import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from monet.tpr import StructuralLobe
import os

class MonetConfig(PretrainedConfig):
    model_type = "monet"
    def __init__(self, base_model_id=None, tpr_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.base_model_id = base_model_id
        self.tpr_dim = tpr_dim

class MonetModel(nn.Module):
    """
    The Unified Bicameral Model.
    Uses forward hooks to implement per-layer parallel experts:
    Output = Neural(X) + Gate * Structural(X)
    """
    def __init__(self, base_model, lobes, gates):
        super().__init__()
        self.base_model = base_model # System 1 (Neural)
        self.lobes = nn.ModuleList(lobes) # System 2 (Structural)
        self.gates = nn.ParameterList(gates) # Corpus Callosum (Gating)
        self.system_mode = "hybrid" # "hybrid", "neural", "structural"
        
        # --- Persistent Structural State Cache ---
        # Stores the hidden state of SSM experts per layer
        self.structural_state_cache = [None for _ in range(len(lobes))]
        
        # Register hooks to integrate System 2 into the Neural stack
        self._register_expert_hooks()
        
    def reset_structural_cache(self):
        """Clears the recurrent memory for a new sequence."""
        self.structural_state_cache = [None for _ in range(len(self.lobes))]
        
    def _register_expert_hooks(self):
        def make_hook(layer_idx, lobe, gate):
            def bicameral_hook(module, layer_input, layer_output):
                # 1. Isolate the Neural Update (System 1)
                neural_update = layer_output[0] - layer_input[0]
                
                # 2. Run Structural Expert (System 2)
                # We retrieve and update the persistent state for this layer
                prev_state = self.structural_state_cache[layer_idx]
                struct_update, next_state = lobe(layer_input[0], state=prev_state)
                
                # Update Cache
                self.structural_state_cache[layer_idx] = next_state
                
                # 3. Mode-Based Integration
                g = torch.sigmoid(gate) 
                
                if self.system_mode == "neural":
                    integrated_update = neural_update
                elif self.system_mode == "structural":
                    integrated_update = 2.0 * struct_update # Use 2x scale for pure mode
                else:
                    # Default: (1-G)*Neural + G*(2*Structural)
                    # We keep the 2.0x scale to compensate for the dropout training
                    integrated_update = (1.0 - g) * neural_update + g * (2.0 * struct_update)
                
                if isinstance(layer_output, tuple):
                    return (layer_input[0] + integrated_update,) + layer_output[1:]
                else:
                    return layer_input[0] + integrated_update
            return bicameral_hook

        for i, layer in enumerate(self.base_model.model.layers):
            layer.register_forward_hook(make_hook(i, self.lobes[i], self.gates[i]))

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, num_thinking_passes=1, **kwargs):
        """
        Standard forward pass, but with optional Global Recurrence and Structural Caching.
        """
        # Always reset the structural cache at the start of a NEW sequence pass.
        # This ensures the SSM (System 2) sees a fresh sequence context.
        self.reset_structural_cache()
        
        # 1. Initial Pass (Embedding -> Stack -> Refined State)
        outputs = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = outputs.hidden_states[-1] # State after 18 layers + Sys2 hooks
        
        # 2. Global Recurrence (Thinking Passes 2...K)
        # We feed the hidden state back into the layer stack
        for k in range(1, num_thinking_passes):
            # We use inputs_embeds to skip the embedding step and reuse the hidden state
            # We must pass the mask and other context to maintain consistency
            outputs = self.base_model(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
            hidden_states = outputs.hidden_states[-1]
            
        # 3. Final Head (Logits)
        # If we did K passes, we project the final refined state
        logits = self.base_model.lm_head(hidden_states)
        
        return logits

def graft_monet(base_model_path, save_path):
    print(f"🏥 Starting Bicameral Surgery on {base_model_path}...")
    
    # 1. Load System 1 (Gemma)
    print("   Loading System 1 (Gemma)...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)
    
    # 2. Initialize System 2 (Structural Hemispheres)
    print("   Initializing System 2 (18 Structural Experts)...")
    hidden_size = base_model.config.hidden_size # 640
    num_layers = len(base_model.model.layers)
    
    lobes = []
    # Gates: Learnable scalars, initialized to -10.0 (Closed/Silent)
    # Sigmoid(-10) is approx 0.0, ensuring we start with the fluent base model.
    gates = [nn.Parameter(torch.tensor(-10.0, dtype=torch.float32)) for _ in range(num_layers)]
    for i in range(num_layers):
        lobe = StructuralLobe(d_model=hidden_size, tpr_dim=hidden_size).to(torch.float32)
        lobes.append(lobe)
    
    # 3. Create Hybrid
    print("   Grafting hemispheres into Unified substrate...")
    monet = MonetModel(base_model, lobes, gates)
    
    # 3.1 Load Pre-trained Nous Weights into EVERY Lobe
    nous_path = "nous/exports/nous_v1.pt"
    if os.path.exists(nous_path):
        print(f"   Integrating Symbolic Intelligence into 18-layer stack...")
        checkpoint = torch.load(nous_path, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        for lobe in monet.lobes:
            lobe.nous.load_state_dict(state_dict)
    else:
        print("   ⚠️ Warning: No pre-trained Nous weights found.")
    
    # 4. Save
    print(f"   Saving Hybrid Model directly to {save_path}...")
    # NOTE: Since MonetGemma isn't a Transformers subclass, we save state_dict for now
    # In production, we'd register it as a custom AutoModel
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    torch.save(monet.state_dict(), os.path.join(save_path, "monet_model.pt"))
    print("✅ Surgery Successful!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="pruned_gemma_3_270m")
    parser.add_argument("--save", type=str, default="monet_v1")
    args = parser.parse_args()
    
    graft_monet(args.base, args.save)
