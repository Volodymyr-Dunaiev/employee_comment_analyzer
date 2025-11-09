# """Streamlit user interface for the comment classifier."""
import os
from io import BytesIO
from typing import Optional, BinaryIO, List
from pathlib import Path
import tempfile
import streamlit as st
from src.core.pipeline import run_inference
from src.core.batch_processor import BatchProcessor
from src.core.classifier import CommentClassifier
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
        1. **Choose training mode**: New model or refine existing
        2. **Upload** your labeled Excel/CSV file
        3. **Configure** column names and parameters (auto-adjusted for refinement)
        4. **Start training** and monitor progress
        5. **Update config.yaml** to use the new model
        
        ### Training Modes
        
        **🆕 Train from base model:**
        - Start fresh with pre-trained xlm-roberta-base or similar
        - Best for: First-time training, completely new categories, poor existing model
        - Epochs: 3-5 | Learning rate: 2e-5
        
        **🔄 Continue training existing model (refinement):**
        - Loads your existing trained model and improves it with new data
        - Best for: Adding new samples, fixing mistakes, expanding categories
        - Epochs: 1-2 | Learning rate: 1e-5 (lower to preserve knowledge)
        - ⚠️ Previous model auto-backed up to `model_backups/v#/`
        
        ### Parameter Guide
        
        | Parameter | New Model | Refinement | What it does |
        |-----------|-----------|------------|--------------|
        | **Epochs** | 3-5 | 1-2 | Training passes. Lower for refinement to avoid overfitting. |
        | **Batch Size** | 8 (4GB RAM)<br>16 (8GB+) | Same | Samples per step. Higher = faster but more memory. |
        | **Learning Rate** | 2e-5 | 1e-5 | Speed of learning. Lower for refinement preserves existing knowledge. |
        | **Test Split** | 10% | 10% | Data reserved for final testing. |
        | **Validation Split** | 10% | 10% | Data for preventing overfitting. |
        
        ### Data Requirements
        - **Minimum:** 50 total samples, 5 per category
        - **Good:** 100+ samples per category
        - **Excellent:** 500+ samples per category
        
        ### Tips
        - UI auto-adjusts parameters when you select refinement mode
        - Monitor validation metrics - if training improves but validation doesn't, you're overfitting
        - Model backups saved automatically before overwriting
        """)
    
    st.markdown("""
    ### Training Instructions
    1. Upload a labeled dataset (Excel/CSV file)
    2. Choose training mode (new or refinement)
    3. Configure training parameters (auto-adjusted for mode)
    4. Start training and monitor progress
    5. Model saved automatically
    
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
            
            # Adjust default epochs based on training mode
            default_epochs = 2 if training_mode == "Continue training existing model (refinement)" else 3
            epochs = st.slider(
                "Number of Epochs", 
                min_value=1, 
                max_value=10, 
                value=default_epochs,
                help="**New model**: 3-5 epochs. **Refinement**: 1-2 epochs to avoid overfitting on new data."
            )
            
            batch_size = st.slider(
                "Batch Size", 
                min_value=4, 
                max_value=32, 
                value=8, 
                step=4,
                help="Number of samples processed together. Larger = faster but uses more memory. Use 4-8 for 4GB RAM, 16-32 for 16GB+ RAM."
            )
            
            # Adjust default learning rate for refinement
            default_lr = 1e-5 if training_mode == "Continue training existing model (refinement)" else 2e-5
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 2e-5, 3e-5, 5e-5],
                value=default_lr,
                format_func=lambda x: f"{x:.0e}",
                help="**New training**: 2e-5 (default). **Refinement**: 1e-5 (lower to preserve existing knowledge)."
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
        
        # Training mode selection
        training_mode = st.radio(
            "Training Mode",
            ["Train from base model", "Continue training existing model (refinement)"],
            index=0,
            help="**Train from base**: Start fresh with a pre-trained model like xlm-roberta-base.\n**Continue training**: Load your existing trained model and refine it with new data."
        )
        
        if training_mode == "Train from base model":
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
        else:  # Continue training existing model
            model_name = st.text_input(
                "Existing Model Path",
                value="./model/ukr_multilabel",
                help="Path to your existing trained model directory. Previous version will be auto-backed up to model_backups/v#/"
            )
            output_dir = model_name  # Save back to same location
            
            # Show warning about refinement
            st.info("💡 **Refinement Tips:**\n"
                   "- Use 1-2 epochs (not 3-5) to avoid overfitting\n"
                   "- Learning rate 1e-5 works better for fine-tuning\n"
                   "- Previous model auto-backed up before overwriting")
            
            # Auto-adjust recommended parameters for refinement
            if 'epochs' not in st.session_state:
                st.session_state.epochs_refinement = True
        
        output_dir_display = st.text_input(
            "Output Directory" if training_mode == "Train from base model" else "Model will be saved to",
            value=output_dir,
            disabled=(training_mode != "Train from base model"),
            help="Model save location"
        ) if training_mode != "Train from base model" else None
        
        # Start training button
        if st.button("Start Training", type="primary"):
            try:
                from src.core.train_interface import train_from_ui, validate_training_data
                import pandas as pd
                import os
                
                # Check if refining an existing model
                is_refinement = (training_mode == "Continue training existing model (refinement)")
                
                # Validate existing model path if refinement mode
                if is_refinement:
                    if not os.path.exists(model_name):
                        st.error(f"❌ Model path not found: `{model_name}`\n\nPlease ensure the model directory exists and contains trained model files.")
                        st.stop()
                    
                    # Check for model files
                    required_files = ['config.json', 'pytorch_model.bin']
                    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_name, f))]
                    if missing_files:
                        st.error(f"❌ Invalid model directory. Missing files: {', '.join(missing_files)}")
                        st.stop()
                    
                    st.info(f"🔄 **Refinement Mode**: Loading existing model from `{model_name}`")
                
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
                
                # Confirm training with mode-specific message
                if is_refinement:
                    st.warning("⚠️ **Refinement Training** will start. Your existing model will be backed up automatically before updating.")
                else:
                    st.warning("⚠️ **New Model Training** will start. This may take several minutes to hours depending on dataset size.")
                
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
        
        **Single File:**
        1. Upload an Excel file containing comments
        2. Click 'Run Classification' to process
        3. Download the results
        
        **Batch Processing:**
        1. Upload multiple Excel/CSV files at once
        2. Click 'Process Batch' to classify all files
        3. Download individual results or combined output
        """)
    
    # Processing mode selection
    processing_mode = st.radio(
        "Processing Mode",
        ["Single File", "Batch Processing"],
        horizontal=True,
        help="Choose single file for one file or batch for multiple files at once"
    )
    
    if processing_mode == "Single File":
        # Single file upload with validation
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
    
    else:  # Batch Processing
        st.subheader("📦 Batch File Processing")
        
        # Initialize session state for batch settings
        if 'batch_combine_results' not in st.session_state:
            st.session_state.batch_combine_results = True
        if 'batch_output_prefix' not in st.session_state:
            st.session_state.batch_output_prefix = "classified_"
        
        # Batch upload
        uploaded_files = st.file_uploader(
            "Upload multiple files",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=True,
            help="Select multiple Excel or CSV files to process at once"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            
            # Show file list
            with st.expander("View uploaded files"):
                for i, file in enumerate(uploaded_files, 1):
                    file.seek(0, os.SEEK_END)
                    size_mb = file.tell() / (1024 * 1024)
                    file.seek(0)
                    st.write(f"{i}. {file.name} ({size_mb:.2f} MB)")
            
            # Batch processing options
            col1, col2 = st.columns(2)
            with col1:
                # Check if using GPU - must use max_workers=1 for CUDA
                device = config['model'].get('device', 'cpu')
                is_gpu = device.lower() in ['cuda', 'gpu']
                
                # Initialize session state for max_workers if not set
                if 'batch_max_workers' not in st.session_state:
                    st.session_state.batch_max_workers = 1
                if 'batch_use_multiprocessing' not in st.session_state:
                    st.session_state.batch_use_multiprocessing = False
                
                if is_gpu:
                    max_workers = 1
                    use_multiprocessing = False
                    st.info("🔒 **GPU/CUDA Mode**")
                    st.caption("Concurrent processing disabled for thread-safety. PyTorch models cannot safely run concurrent inference on GPU.")
                    # Display disabled input to show the value
                    st.number_input(
                        "Concurrent files",
                        min_value=1,
                        max_value=1,
                        value=1,
                        disabled=True,
                        help="GPU mode requires single-threaded processing"
                    )
                else:
                    # Multiprocessing toggle
                    use_multiprocessing = st.checkbox(
                        "Use multiprocessing (true parallelism)",
                        value=st.session_state.batch_use_multiprocessing,
                        help="Creates separate processes with isolated models for true parallel processing. Higher memory usage but better CPU utilization."
                    )
                    st.session_state.batch_use_multiprocessing = use_multiprocessing
                    
                    if use_multiprocessing:
                        st.caption("✨ **Multiprocessing Mode** — True parallel processing with separate model instances")
                        max_workers = st.number_input(
                            "Concurrent processes",
                            min_value=1,
                            max_value=8,
                            value=st.session_state.batch_max_workers,
                            key='max_workers_input',
                            help="Number of processes to run in parallel. Each process loads its own model instance."
                        )
                    else:
                        st.caption("ℹ️ **Threading Mode (Recommended for small batches)** — PyTorch inference stays single-threaded (only I/O is parallelized)")
                        max_workers = st.number_input(
                            "Concurrent files",
                            min_value=1,
                            max_value=5,
                            value=st.session_state.batch_max_workers,
                            key='max_workers_input',
                            help="Number of files to process simultaneously. Values >1 only speed up file reading/writing, not classification."
                        )
                    # Update session state with user choice
                    st.session_state.batch_max_workers = max_workers
            with col2:
                combine_results = st.checkbox(
                    "Create combined output",
                    value=st.session_state.batch_combine_results,
                    help="Generate a single file with all results combined"
                )
                st.session_state.batch_combine_results = combine_results
                
                output_prefix = st.text_input(
                    "Output filename prefix",
                    value=st.session_state.batch_output_prefix,
                    help="Prefix to add to output filenames"
                )
                st.session_state.batch_output_prefix = output_prefix
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                dry_run_clicked = st.button("🔍 Dry Run (Validate Only)", help="Check files without processing")
            with col_btn2:
                process_clicked = st.button("Process Batch", type="primary")
            
            # Dry run validation
            if dry_run_clicked:
                try:
                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        
                        # Save uploaded files temporarily
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = temp_path / uploaded_file.name
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getvalue())
                            file_paths.append(file_path)
                        
                        # Initialize batch processor (no classifier needed for dry-run)
                        config = load_config()
                        classifier = CommentClassifier(config)
                        processor = BatchProcessor(
                            classifier,
                            max_workers=1,
                            text_column=text_column
                        )
                        
                        # Run dry-run validation
                        with st.spinner("Validating files..."):
                            validation_results = processor.dry_run_validation(file_paths)
                        
                        # Display results
                        st.subheader("📋 Validation Report")
                        
                        for filename, report in validation_results.items():
                            status_icon = {"valid": "✅", "warning": "⚠️", "error": "❌"}[report['status']]
                            
                            with st.expander(f"{status_icon} {filename}"):
                                if report['error']:
                                    st.error(f"Error: {report['error']}")
                                else:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Rows", report['total_rows'])
                                        st.metric("Empty Rate", f"{report['empty_rate']*100:.1f}%")
                                    with col2:
                                        st.metric("Avg Text Length", report['avg_text_length'])
                                        st.metric("Max Text Length", report['max_text_length'])
                                    with col3:
                                        st.metric("Encoding", report['encoding'])
                                        st.metric("Est. Memory", f"{report['estimated_memory_mb']} MB")
                                    
                                    st.write("**Issues:**")
                                    for issue in report['issues']:
                                        st.write(f"- {issue}")
                        
                        # Summary
                        valid_count = sum(1 for r in validation_results.values() if r['status'] == 'valid')
                        warning_count = sum(1 for r in validation_results.values() if r['status'] == 'warning')
                        error_count = sum(1 for r in validation_results.values() if r['status'] == 'error')
                        
                        if error_count > 0:
                            st.error(f"❌ {error_count} file(s) have errors")
                        elif warning_count > 0:
                            st.warning(f"⚠️ {warning_count} file(s) have warnings, {valid_count} are ready")
                        else:
                            st.success(f"✅ All {valid_count} files are ready for processing")
                        
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
                    logger.error(f"Dry-run validation error: {str(e)}", exc_info=True)
            
            # Process button
            if process_clicked:
                try:
                    # Create temporary directory for processing
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        
                        # Save uploaded files temporarily
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = temp_path / uploaded_file.name
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getvalue())
                            file_paths.append(file_path)
                        
                        # Initialize batch processor
                        config = load_config()
                        classifier = CommentClassifier(config)
                        processor = BatchProcessor(
                            classifier,
                            max_workers=max_workers,
                            text_column=text_column,
                            use_multiprocessing=use_multiprocessing
                        )
                        
                        # Validate files and cache loaded data
                        with st.spinner("Validating files..."):
                            valid_files, errors, cached_data = processor.validate_files(file_paths)
                        
                        if errors:
                            st.warning("⚠️ Some files have issues:")
                            for error in errors:
                                st.write(f"- {error}")
                        
                        if not valid_files:
                            st.error("No valid files to process!")
                        else:
                            # Setup progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(filename: str, current: int, total: int):
                                """Callback for progress updates."""
                                progress = current / total if total > 0 else 0
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {filename} ({current}/{total})")
                            
                            # Process files with progress tracking
                            result = processor.process_files(
                                valid_files,
                                output_dir=temp_path / "results",
                                output_prefix=output_prefix,
                                combine_results=combine_results,
                                validated_data=cached_data,
                                progress_callback=update_progress
                            )
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Display results summary
                            st.success("✅ Batch processing complete!")
                            
                            summary = result.get_summary()
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Files", summary['total_files'])
                            with col2:
                                st.metric("Successful", summary['successful_files'])
                            with col3:
                                st.metric("Failed", summary['failed_files'])
                            with col4:
                                st.metric("Rows Processed", summary['processed_comments'])
                            
                            st.write(f"⏱️ Processing time: {summary['processing_time']}")
                            st.write(f"📊 Success rate: {summary['success_rate']}")
                            
                            # Show warning if any comments were skipped with per-file breakdown
                            if summary['skipped_comments'] > 0:
                                total_rows = summary['total_rows']
                                skip_pct = (summary['skipped_comments'] / total_rows * 100) if total_rows > 0 else 0
                                
                                warning_msg = (
                                    f"⚠️ **{summary['skipped_comments']} empty or invalid comments were skipped "
                                    f"({skip_pct:.1f}% of {total_rows} total rows).** "
                                    "These rows have `is_skipped=True` and zero probabilities."
                                )
                                
                                # Add per-file breakdown
                                files_with_skips = {
                                    fname: count for fname, count in result.skip_counts_by_file.items() if count > 0
                                }
                                if files_with_skips:
                                    warning_msg += "\n\n**Per-file breakdown:**\n"
                                    for fname, count in sorted(files_with_skips.items(), key=lambda x: x[1], reverse=True):
                                        # Get total rows for this file
                                        file_df = result.results_by_file.get(fname)
                                        file_total = len(file_df) if file_df is not None else count
                                        file_pct = (count / file_total * 100) if file_total > 0 else 0
                                        warning_msg += f"\n- `{fname}`: {count} skipped ({file_pct:.1f}%)"
                                
                                st.warning(warning_msg)
                            
                            # Show failures if any
                            if result.failed_files:
                                with st.expander("❌ Failed Files"):
                                    for filename, error in result.failed_files:
                                        st.write(f"**{filename}:** {error}")
                            
                            # Provide downloads
                            st.subheader("📥 Download Results")
                            
                            # Individual file downloads
                            if result.successful_files:
                                st.write("**Individual Results:**")
                                for filename in result.successful_files:
                                    output_path = temp_path / "results" / f"classified_{filename}"
                                    if output_path.exists():
                                        with open(output_path, 'rb') as f:
                                            st.download_button(
                                                f"📄 {filename}",
                                                data=f.read(),
                                                file_name=f"classified_{filename}",
                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                key=f"download_{filename}"
                                            )
                            
                            # Combined output download
                            if combine_results and result.combined_output_path is not None:
                                st.write("**Combined Output:**")
                                with open(result.combined_output_path, 'rb') as f:
                                    st.download_button(
                                        "📦 Download All (Combined)",
                                        data=f.read(),
                                        file_name=result.combined_output_path.name,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="download_combined"
                                    )
                
                except Exception as e:
                    display_error(f"Batch processing error: {str(e)}")
                    logger.exception("Batch processing failed")

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
