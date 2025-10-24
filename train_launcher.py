"""
Training launcher for standalone executable.
Provides a simple command-line interface for training models on any laptop.
"""
import os
import sys
import argparse

def main():
    # Set working directory to exe location
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.chdir(app_dir)
    
    print("=" * 70)
    print("Ukrainian Comments Classifier - Model Training Utility")
    print("=" * 70)
    print(f"Working directory: {app_dir}")
    print()
    
    parser = argparse.ArgumentParser(
        description="Train or refine the classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick training (1 epoch):
    Train_Model.exe --data data\\train.xlsx --epochs 1
  
  Full training:
    Train_Model.exe --data data\\train.xlsx --epochs 3 --batch_size 8
  
  Refine existing model:
    Train_Model.exe --data data\\new_samples.xlsx --model_name_or_path model\\ukr_multilabel --epochs 2

Training data format:
  Your Excel file can have:
  - Column 'text' with comments + column 'labels' with categories
  - OR: 'text' column + 'Category 1', 'Category 2', etc. with category names
        """
    )
    
    parser.add_argument('--data', required=True, 
                       help='Path to training Excel/CSV file (e.g., data\\train.xlsx)')
    parser.add_argument('--model_name_or_path', default='xlm-roberta-base',
                       help='Base model to start from (default: xlm-roberta-base)')
    parser.add_argument('--model_dir', default='model\\ukr_multilabel',
                       help='Where to save the trained model (default: model\\ukr_multilabel)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training (default: 8)')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--text_column', default='text',
                       help='Name of text column in your file (default: text)')
    parser.add_argument('--labels_column', default='labels',
                       help='Name of labels column (default: labels, or auto-detect Category columns)')
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not os.path.exists(args.data):
        print(f"ERROR: Training data file not found: {args.data}")
        print()
        print("Please ensure your training file exists.")
        print("Example path: data\\train.xlsx")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print(f"Training data: {args.data}")
    print(f"Base model: {args.model_name_or_path}")
    print(f"Output directory: {args.model_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    try:
        # Import training function
        from src.core.train import train
        
        print("Starting training...")
        print("=" * 70)
        print()
        
        result = train(
            data_path=args.data,
            model_name_or_path=args.model_name_or_path,
            model_dir=args.model_dir,
            text_column=args.text_column,
            labels_column=args.labels_column,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=42,
            threshold=0.5,
            fp16=False,
            use_onnx_export=False
        )
        
        print()
        print("=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
        print(f"Model saved to: {result['model_dir']}")
        print()
        print("Test metrics:")
        for key, value in result.get('test_metrics', {}).items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        print()
        print("To use this model in the app:")
        print(f"  1. Update config.yaml:")
        print(f"     model.path: {args.model_dir}")
        print("  2. Restart the Comments_Classifier.exe")
        print()
        
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR: Training failed")
        print("=" * 70)
        print(f"{type(e).__name__}: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

if __name__ == '__main__':
    main()
