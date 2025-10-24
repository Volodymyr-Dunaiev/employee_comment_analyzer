# """Streamlit user interface for the comment classifier."""
import os
from io import BytesIO
from typing import Optional, BinaryIO
import streamlit as st
from src.core.pipeline import run_inference
from src.core.errors import PipelineError
from src.utils.logger import get_logger
from src.utils.config import load_config, ConfigError

logger = get_logger(__name__)

def validate_file(file: Optional[BinaryIO]) -> bool:
    # Validate the uploaded file.
    #
    # Args:
    #     file: The uploaded file object
    #
    # Returns:
    #     bool: True if file is valid
    
    if not file:
        return False
        
    config = load_config()
    max_size = config['security']['max_file_size_mb'] * 1024 * 1024
    allowed_extensions = config['security']['allowed_extensions']
    
    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > max_size:
        st.error(f"File size exceeds maximum limit of {config['security']['max_file_size_mb']}MB")
        return False
        
    # Check extension
    ext = os.path.splitext(file.name)[1][1:].lower()
    if ext not in allowed_extensions:
        st.error(f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}")
        return False
        
    return True

def display_error(message: str):
    # Display error messages consistently.
    st.error(message)
    logger.error(message)

def show_training_tab(config):
    # Display the training interface.
    st.header("Model Training")
    
    # Add helpful information box
    with st.expander("ℹ️ Training Guide - Click to expand", expanded=False):
        st.markdown("""
        ### Quick Start
        1. **Upload** your labeled Excel/CSV file
        2. **Configure** column names and parameters
        3. **Start training** and monitor progress
        4. **Update config.yaml** to use the new model
        
        ### Parameter Guide
        
        | Parameter | Recommended | What it does |
        |-----------|------------|--------------|
        | **Epochs** | 3-5 | How many times to train on full dataset. More = better learning, but diminishing returns after 5. |
        | **Batch Size** | 8 (4GB RAM)<br>16 (8GB RAM)<br>32 (16GB+ RAM) | Samples per training step. Higher = faster but needs more memory. |
        | **Learning Rate** | 2e-5 | Speed of learning. Use 1e-5 for fine-tuning, 2e-5 for new training, 5e-5 for quick experiments. |
        | **Test Split** | 10% | Data reserved for final evaluation (never seen during training). |
        | **Validation Split** | 10% | Data used to prevent overfitting during training. |
        
        ### Data Requirements
        - **Minimum:** 50 total samples, 5 per category
        - **Good:** 100+ samples per category
        - **Excellent:** 500+ samples per category
        
        ### Tips
        - Start with default parameters (3 epochs, batch size 8, learning rate 2e-5)
        - Monitor validation metrics - if training improves but validation doesn't, you're overfitting
        - Previous models are auto-backed up to `model_backups/v#/` before overwriting
        """)
    
    st.markdown("""
    ### Training Instructions
    1. Upload a labeled dataset (Excel/CSV file)
    2. Configure training parameters
    3. Start training process
    4. Monitor training progress
    5. Save the trained model
    
    **Supported Label Formats:**
    - **Single column**: All labels in one cell, separated by commas (e.g., "Category1, Category2")
    - **Multiple columns**: One label per column (e.g., label_1, label_2, label_3 columns)
    """)
    
    # Upload training data file
    training_file = st.file_uploader(
        "Upload Training Data (Excel/CSV)", 
        type=["xlsx", "xls", "csv"],
        key="training_file"
    )
    
    if training_file:
        st.success(f"File uploaded: {training_file.name}")
        
        # Create two-column layout for configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Configuration")
            text_column = st.text_input("Text Column Name", value=config['data'].get('text_column', 'text'))
            
            # Label format selection
            label_format = st.radio(
                "Label Format",
                ["Single column (comma-separated)", "Multiple columns (one label per column)"],
                help="Choose how labels are organized in your data"
            )
            
            if label_format == "Single column (comma-separated)":
                labels_column = st.text_input("Labels Column Name", value="labels")
            else:
                labels_column = st.text_input(
                    "Label Column Pattern", 
                    value="label_*",
                    help="Enter column prefix (e.g., 'label_*', 'category_*', 'cat_*')"
                )
            
        with col2:
            st.subheader("Training Parameters")
            
            epochs = st.slider(
                "Number of Epochs", 
                min_value=1, 
                max_value=10, 
                value=3,
                help="How many times the model will see the entire training dataset. More epochs = better learning but risk of overfitting. Start with 3."
            )
            
            batch_size = st.slider(
                "Batch Size", 
                min_value=4, 
                max_value=32, 
                value=8, 
                step=4,
                help="Number of samples processed together. Larger = faster but uses more memory. Use 4-8 for 4GB RAM, 16-32 for 16GB+ RAM."
            )
            
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 2e-5, 3e-5, 5e-5],
                value=2e-5,
                format_func=lambda x: f"{x:.0e}",
                help="How fast the model learns. 2e-5 is a good default. Lower (1e-5) for fine-tuning existing models, higher (5e-5) for new training."
            )
            
        # Configure train/test/validation split percentages
        col3, col4 = st.columns(2)
        with col3:
            test_size = st.slider(
                "Test Split (%)", 
                min_value=5, 
                max_value=30, 
                value=10,
                help="Percentage of data reserved for final testing (never seen during training). Recommended: 10%."
            )
        with col4:
            valid_size = st.slider(
                "Validation Split (%)", 
                min_value=5, 
                max_value=30, 
                value=10,
                help="Percentage of data for validation during training (used to prevent overfitting). Recommended: 10%."
            )
        
        # Model architecture selection
        st.subheader("Model Configuration")
        model_name = st.selectbox(
            "Base Model",
            ["xlm-roberta-base", "bert-base-multilingual-cased", "xlm-roberta-large"],
            index=0,
            help="Pre-trained model to start from. xlm-roberta-base (default) is best for Ukrainian text. Large version is slower but more accurate."
        )
        
        output_dir = st.text_input(
            "Output Directory", 
            value="./trained_model",
            help="Where to save the trained model files. Default: ./trained_model"
        )
        
        # Start training button
        if st.button("Start Training", type="primary"):
            try:
                from src.core.train_interface import train_from_ui, validate_training_data
                import pandas as pd
                
                # Load and validate training data
                with st.spinner("Loading training data..."):
                    # Read the file based on extension
                    if training_file.name.endswith('.csv'):
                        df = pd.read_csv(training_file)
                    else:
                        df = pd.read_excel(training_file)
                    
                    # Validate data
                    is_valid, error_msg = validate_training_data(df, text_column, labels_column)
                    
                    if not is_valid:
                        st.error(f"❌ Validation Error: {error_msg}")
                        st.stop()
                    
                    st.success(f"✅ Loaded {len(df)} training samples")
                
                # Show data preview
                with st.expander("Preview Training Data"):
                    st.dataframe(df.head(10))
                
                # Confirm training
                st.warning("⚠️ Training will start. This may take several minutes to hours depending on dataset size.")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(status: str, progress: float):
                    progress_bar.progress(progress)
                    status_text.text(status)
                
                # Execute training pipeline
                with st.spinner("Training in progress..."):
                    training_file.seek(0)  # Reset file pointer to beginning
                    results = train_from_ui(
                        training_file=training_file,
                        text_column=text_column,
                        labels_column=labels_column,
                        model_name=model_name,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        test_size=test_size/100,
                        valid_size=valid_size/100,
                        output_dir=output_dir,
                        progress_callback=update_progress
                    )
                
                # Show training completion and results
                st.success("🎉 Training completed successfully!")
                
                # Display dataset split statistics
                st.subheader("📊 Dataset Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", results['num_samples'])
                with col2:
                    st.metric("Training", results['train_samples'])
                with col3:
                    st.metric("Validation", results['valid_samples'])
                with col4:
                    st.metric("Test", results['test_samples'])
                
                # Training metrics
                st.subheader("📈 Training Results")
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("Training Loss", f"{results['train_loss']:.4f}")
                with col6:
                    st.metric("Number of Labels", results['num_labels'])
                
                # Display performance metrics from test set evaluation
                if 'test_metrics' in results:
                    st.subheader("🎯 Test Set Performance")
                    metrics = results['test_metrics']
                    
                    col7, col8, col9, col10 = st.columns(4)
                    with col7:
                        if 'eval_micro_f1' in metrics:
                            st.metric("Micro F1", f"{metrics['eval_micro_f1']:.4f}")
                    with col8:
                        if 'eval_macro_f1' in metrics:
                            st.metric("Macro F1", f"{metrics['eval_macro_f1']:.4f}")
                    with col9:
                        if 'eval_micro_precision' in metrics:
                            st.metric("Precision", f"{metrics['eval_micro_precision']:.4f}")
                    with col10:
                        if 'eval_micro_recall' in metrics:
                            st.metric("Recall", f"{metrics['eval_micro_recall']:.4f}")
                
                st.success(f"✅ Model saved to: `{results['model_path']}`")
                
                # Show labels
                with st.expander("📋 Trained Labels"):
                    for i, label in enumerate(results['labels'], 1):
                        st.write(f"{i}. {label}")
                
                st.info("""
                ### 🚀 Next Steps:
                1. Update `config.yaml` to use your trained model:
                   ```yaml
                   model:
                     path: "{}"
                   ```
                2. Switch to the Classification tab to test your model
                3. Monitor performance and retrain if needed
                """.format(results['model_path']))
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                logger.exception("Training error")
                st.info("""
                ### Troubleshooting:
                - Check that your data has the correct column names
                - Ensure labels are in the correct format (list or comma-separated)
                - Verify you have enough disk space for the model
                - Check the logs for more details
                """)

def show_inference_tab(config):
    # Display the inference interface.
    st.header("Comment Classification")
    
    categories = config['categories']
    text_column = config['data']['text_column']
    
    # Sidebar for configuration display
    with st.sidebar:
        st.header("Configuration")
        st.write("Text column:", text_column)
        st.write("Categories:")
        for cat in categories:
            st.write(f"- {cat}")

        st.markdown("""
        ### Instructions
        1. Upload an Excel file containing comments
        2. Click 'Run Classification' to process
        3. Download the results
        """)

    # File upload with validation
    input_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    if input_file and validate_file(input_file):
        # In-memory file handling
        output_file = BytesIO()

        # Progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)

        def update_progress(current: int, total: int) -> None:
            percent = int((current / total) * 100)
            progress_bar.progress(percent)
            progress_text.text(f"Processing... {percent}%")

        # Process button with error handling
        if st.button("Run Classification", type="primary"):
            try:
                with st.spinner("Processing..."):
                    run_inference(
                        input_file,
                        output_file,
                        text_column,
                        categories,
                        update_progress
                    )

                st.success("Classification completed successfully!")

                # Offer download
                st.download_button(
                    "Download Results",
                    data=output_file.getvalue(),
                    file_name="classified_comments.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except PipelineError as e:
                display_error(f"Processing error: {str(e)}")
            except Exception as e:
                display_error("An unexpected error occurred")

def main():
    # Main application entry point.
    try:
        st.set_page_config(
            page_title="Ukrainian Comment Classifier",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("Ukrainian Comment Classifier")

        # Load configuration
        try:
            config = load_config()
        except ConfigError as e:
            display_error(f"Configuration error: {str(e)}")
            return

        # Create tabs for different functionalities
        tab1, tab2 = st.tabs(["📊 Classification", "🎓 Training"])
        
        with tab1:
            show_inference_tab(config)
            
        with tab2:
            show_training_tab(config)

    except Exception as e:
        logger.exception("Unexpected error in UI")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
