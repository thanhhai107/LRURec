import pickle
import sys
from pathlib import Path

def analyze_dataset(pkl_path):
    """Analyze preprocessed dataset statistics"""
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']
    umap = dataset['umap']
    smap = dataset['smap']
    
    # Calculate statistics
    num_users = len(umap)
    num_items = len(smap)
    
    # Count interactions
    train_interactions = sum(len(items) for items in train.values())
    val_interactions = sum(len(items) for items in val.values())
    test_interactions = sum(len(items) for items in test.values())
    total_interactions = train_interactions + val_interactions + test_interactions
    
    # Calculate sequence length statistics
    train_lengths = [len(items) for items in train.values()]
    avg_train_len = sum(train_lengths) / len(train_lengths) if train_lengths else 0
    min_train_len = min(train_lengths) if train_lengths else 0
    max_train_len = max(train_lengths) if train_lengths else 0
    
    # Density
    density = total_interactions / (num_users * num_items) * 100 if (num_users * num_items) > 0 else 0
    
    # Print results
    print("=" * 80)
    print(f"Dataset: {pkl_path.parent.name}")
    print("=" * 80)
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  • Total Users:        {num_users:,}")
    print(f"  • Total Items:        {num_items:,}")
    print(f"  • Total Interactions: {total_interactions:,}")
    print(f"  • Density:            {density:.4f}%")
    
    print(f"\n🔢 INTERACTIONS BY SPLIT:")
    print(f"  • Train:      {train_interactions:,} ({train_interactions/total_interactions*100:.2f}%)")
    print(f"  • Validation: {val_interactions:,} ({val_interactions/total_interactions*100:.2f}%)")
    print(f"  • Test:       {test_interactions:,} ({test_interactions/total_interactions*100:.2f}%)")
    
    print(f"\n📏 SEQUENCE LENGTH (Train):")
    print(f"  • Average: {avg_train_len:.2f}")
    print(f"  • Min:     {min_train_len}")
    print(f"  • Max:     {max_train_len}")
    
    print(f"\n👥 PER USER AVERAGE:")
    print(f"  • Interactions per user: {total_interactions/num_users:.2f}")
    print(f"  • Train items per user:  {train_interactions/num_users:.2f}")
    
    print(f"\n📦 PER ITEM AVERAGE:")
    print(f"  • Interactions per item: {total_interactions/num_items:.2f}")
    
    print("=" * 80)
    print()


def main():
    # Find all preprocessed datasets
    preprocessed_root = Path('data/preprocessed')
    
    if not preprocessed_root.exists():
        print("❌ Preprocessed folder not found!")
        return
    
    pkl_files = list(preprocessed_root.glob('*/dataset.pkl'))
    
    if not pkl_files:
        print("❌ No preprocessed datasets found!")
        return
    
    print(f"\n🔍 Found {len(pkl_files)} preprocessed dataset(s)\n")
    
    # If specific dataset provided as argument
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        pkl_files = [f for f in pkl_files if dataset_name in f.parent.name]
        if not pkl_files:
            print(f"❌ Dataset '{dataset_name}' not found!")
            return
    
    # Analyze each dataset
    for pkl_path in sorted(pkl_files):
        analyze_dataset(pkl_path)


if __name__ == "__main__":
    main()
