import torch
from monet.graft import MonetModel
from monet.tpr import StructuralLobe
from transformers import AutoModelForCausalLM

def count_params():
    print("📊 Monet Parameter Analysis")
    print("="*40)
    
    # Init architecture
    base_model = AutoModelForCausalLM.from_pretrained("pruned_gemma_3_270m")
    hidden_size = base_model.config.hidden_size
    num_layers = len(base_model.model.layers)
    
    lobes = [StructuralLobe(d_model=hidden_size, tpr_dim=hidden_size) for _ in range(num_layers)]
    gates = [torch.nn.Parameter(torch.tensor(0.01)) for _ in range(num_layers)]
    
    from monet.graft import MonetModel
    monet = MonetModel(base_model, lobes, gates)
    
    total_params = sum(p.numel() for p in monet.parameters())
    base_params = sum(p.numel() for p in monet.base_model.parameters())
    lobe_total_params = sum(sum(p.numel() for p in lobe.parameters()) for lobe in monet.lobes)
    gate_params = sum(p.numel() for p in monet.gates)
    
    print(f"Total Model Size: {total_params / 1e6:.2f}M")
    print(f"System 1 (Neural): {base_params / 1e6:.2f}M ({base_params/total_params*100:.1f}%)")
    print(f"System 2 (Struct): {lobe_total_params / 1e6:.2f}M ({lobe_total_params/total_params*100:.1f}%)")
    
    print("\nLayer Composition (Sample Layer 0):")
    print("-" * 40)
    
    sample_lobe = monet.lobes[0]
    lobe_params = sum(p.numel() for p in sample_lobe.parameters())
    
    lobe_submodules = {
        "Roles (Memory)": sample_lobe.roles,
        "Expert Gating": sample_lobe.expert_gating,
        "Syntax Expert": sample_lobe.syntax_expert,
        "Logic Expert": sample_lobe.logic_expert,
        "Formal Expert": [sample_lobe.to_formal, sample_lobe.from_formal, sample_lobe.nous]
    }
    
    for name, module in lobe_submodules.items():
        if isinstance(module, list):
            count = sum(sum(p.numel() for p in m.parameters()) for m in (module if module[2] else module[:2]))
        elif isinstance(module, torch.nn.Parameter):
            count = module.numel()
        else:
            count = sum(p.numel() for p in module.parameters())
        print(f"{name:<15}: {count / 1e3:.1f}K ({count/lobe_params*100:.1f}% of Lobe)")

if __name__ == "__main__":
    count_params()
