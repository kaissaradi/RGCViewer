import os
from pathlib import Path
from qtpy.QtWidgets import QFileDialog, QMessageBox, QApplication, QStyle
from qtpy.QtCore import QThread
from qtpy.QtGui import QStandardItem, QStandardItemModel, QColor, QIcon
from qtpy.QtCore import Qt

from data_manager import DataManager
from gui.workers import RefinementWorker, SpatialWorker
from gui.widgets import PandasModel
import gui.plotting as plotting

def load_directory(main_window, kilosort_dir=None, dat_file=None):
    """Handles the logic for loading a Kilosort directory."""
    # Set default directory: use /home/localadmin/Documents/Development/data/sorted if it exists, otherwise home
    default_dir = Path("/home/localadmin/Documents/Development/data/sorted")
    if not default_dir.exists():
        default_dir = Path.home()
    if kilosort_dir is None:
        ks_dir_name = QFileDialog.getExistingDirectory(main_window, "Select Kilosort Output Directory", str(default_dir))
    else:
        ks_dir_name = kilosort_dir
    if not ks_dir_name:
        return
        
    main_window.status_bar.showMessage("Loading Kilosort files...")
    QApplication.processEvents()
    
    main_window.data_manager = DataManager(ks_dir_name, main_window)
    success, message = main_window.data_manager.load_kilosort_data()
    
    if not success:
        QMessageBox.critical(main_window, "Loading Error", message)
        main_window.status_bar.showMessage("Loading failed.", 5000)
        return
        
    main_window.status_bar.showMessage("Kilosort files loaded. Please select the raw data file.")
    QApplication.processEvents()
    
    if dat_file is None:
        dat_file, _ = QFileDialog.getOpenFileName(main_window, "Select Raw Data File (.dat or .bin)",
                                              str(main_window.data_manager.dat_path_suggestion.parent),
                                              "Binary Files (*.dat *.bin)")
    if not dat_file:
        main_window.status_bar.showMessage("Data loading cancelled by user.", 5000)
        main_window.analysis_tabs.setTabEnabled(main_window.analysis_tabs.indexOf(main_window.raw_trace_tab), False)
    else:
        # Use the DataManager's set_dat_path method to create the memory map
        main_window.data_manager.set_dat_path(Path(dat_file))
        main_window.analysis_tabs.setTabEnabled(main_window.analysis_tabs.indexOf(main_window.raw_trace_tab), True)
    
    main_window.status_bar.showMessage("Building cluster dataframe...")
    QApplication.processEvents()
    
    main_window.data_manager.build_cluster_dataframe()

    # Automatically check for and load vision files in the same directory.
    # We will search for any set of matching files (.ei, .sta, .params).
    vision_dir = Path(ks_dir_name)
    dataset_name = None
    
    print(f"[DEBUG] Scanning for Vision files in {vision_dir}...")

    # Find any .ei file and check if its siblings (.sta, .params) exist.
    for ei_file in vision_dir.glob('*.ei'):
        base_name = ei_file.stem  # Gets the filename without the extension
        dataset_name = base_name
        print(f"[DEBUG] Found EI file with base name: '{dataset_name}'")
        break  # Use the first EI file found
        
        # Check if the corresponding .sta and .params files exist
        # if (vision_dir / f'{base_name}.sta').exists() and \
        #    (vision_dir / f'{base_name}.params').exists():
            
        #     dataset_name = base_name
        #     print(f"[DEBUG] Found complete Vision file set with base name: '{dataset_name}'")
        #     break  # Found a set, no need to check further

    if dataset_name:
        main_window.status_bar.showMessage(f"Found Vision EI file ('{dataset_name}') in directory - loading automatically...")
        QApplication.processEvents()
        
        # Call the updated method with the discovered dataset name
        success, message = main_window.data_manager.load_vision_data(vision_dir, dataset_name)
        
        print(f"[DEBUG] Vision data load completed. Success: {success}, Message: {message}")
    else:
        print(f"[DEBUG] No complete set of .ei, .sta, and .params files found. Skipping automatic loading.")
    
    # Load cell type file
    ls_txt = list(vision_dir.glob('*.txt'))
    if len(ls_txt) > 0:
        txt_file = ls_txt[0]
    else:
        txt_file = None
    main_window.data_manager.load_cell_type_file(txt_file)

    # Check if a tree structure file exists and load it, otherwise populate with default structure
    tree_file_path = os.path.join(ks_dir_name, 'cluster_group_refined_tree.json')
    if os.path.exists(tree_file_path):
        main_window.data_manager.load_tree_structure(tree_file_path)
    else:
        populate_tree_view(main_window)  # Use the new tree view instead of table view

    start_worker(main_window)
    main_window.central_widget.setEnabled(True)
    main_window.load_vision_action.setEnabled(True)
    main_window.load_classification_action.setEnabled(True)
    main_window.status_bar.showMessage(f"Successfully loaded {len(main_window.data_manager.cluster_df)} clusters.", 5000)

    # Hide sta_panel if no vision STAs are loaded
    if not main_window.data_manager.vision_stas:
        main_window.sta_panel.hide()

def load_vision_directory(main_window):
    """Handles the logic for loading a Vision analysis directory."""
    if not main_window.data_manager:
        QMessageBox.warning(main_window, "No Kilosort Data", "Please load a Kilosort directory first.")
        return

    vision_dir_name = QFileDialog.getExistingDirectory(main_window, "Select Vision Analysis Directory")
    if not vision_dir_name:
        return

    main_window.status_bar.showMessage(f"Loading Vision files from {Path(vision_dir_name).name}...")
    QApplication.processEvents()

    success, message = main_window.data_manager.load_vision_data(vision_dir_name)

    if success:
        main_window.status_bar.showMessage(message, 5000)
        # Trigger a refresh of the currently selected cluster to show new data
        if main_window._get_selected_cluster_id() is not None:
            on_cluster_selection_changed(main_window)
    else:
        # If Vision loading fails but params/sta exist, we can still proceed with available data
        vision_path = Path(vision_dir_name)
        
        # Check if params or sta files exist - if so, we can still display them without EI
        params_path = vision_path / 'sta_params.params'
        sta_path = vision_path / 'sta_container.sta'
        
        if params_path.exists() or sta_path.exists():
            # Still try to load what we can
            success_partial, message_partial = main_window.data_manager.load_vision_data(vision_dir_name)
            if success_partial:
                main_window.status_bar.showMessage(f"Loaded partial Vision data: {message_partial}", 5000)
            else:
                # Show a warning instead of a critical error, since partial data can still be useful
                QMessageBox.warning(main_window, "Vision Loading Warning", 
                    f"Could not load all Vision data, but some files were found. {message}")
            # Still trigger refresh so that any available data is shown
            if main_window._get_selected_cluster_id() is not None:
                on_cluster_selection_changed(main_window)
        else:
            # If no params/sta files exist, show the original error
            QMessageBox.critical(main_window, "Vision Loading Error", message)
            main_window.status_bar.showMessage("Vision loading failed.", 5000)

def on_cluster_selection_changed(main_window):
    """
    Handles a cluster selection by triggering the main window's selection timer.
    """
    cluster_id = main_window._get_selected_cluster_id()
    if cluster_id is None:
        return
    
    # This single call now handles the debouncing logic.
    main_window.update_cluster_views(cluster_id)

# In gui/callbacks.py

def on_tab_changed(main_window, index):
    """Handles logic when the user switches between analysis tabs."""

    # --- FIX: Aggressively disconnect the zoom signal to prevent it from firing on hide/resize.
    # It's safe to call disconnect even if it's not connected.
    try:
        main_window.raw_trace_plot.sigXRangeChanged.disconnect(main_window.on_raw_trace_zoom)
    except (TypeError, RuntimeError):
        # This can happen if it was already disconnected, which is fine.
        pass

    current_widget = main_window.analysis_tabs.widget(index)
    cluster_id = main_window._get_selected_cluster_id()
    if cluster_id is None:
        return

    # Handle Raw Trace tab
    if current_widget == main_window.raw_trace_tab:
        # --- FIX: Reconnect the zoom signal ONLY when this tab is active.
        main_window.raw_trace_plot.sigXRangeChanged.connect(main_window.on_raw_trace_zoom)
        main_window.load_raw_trace_data(cluster_id)

    # Handle Spatial Analysis tab
    elif current_widget == main_window.summary_tab:
        plotting.draw_summary_EI_plot(main_window, cluster_id)
        # The rest of your logic for this tab remains the same...
        has_vision_ei = main_window.data_manager.vision_eis and (cluster_id + 1) in main_window.data_manager.vision_eis
        if not has_vision_ei and cluster_id not in main_window.data_manager.heavyweight_cache:
            main_window.status_bar.showMessage(f"Requesting spatial analysis for C{cluster_id}...", 3000)
            main_window.summary_canvas.fig.clear()
            main_window.summary_canvas.fig.text(0.5, 0.5, f"Loading C{cluster_id}...", ha='center', va='center', color='white')
            main_window.summary_canvas.draw()
            QApplication.processEvents()
            if main_window.spatial_worker:
                main_window.spatial_worker.add_to_queue(cluster_id, high_priority=True)

    # Handle STA Analysis tab
    elif current_widget == main_window.sta_tab:
        view_type = getattr(main_window, 'current_sta_view', 'rf')
        if view_type == "rf":
            plotting.draw_sta_plot(main_window, cluster_id)
        elif view_type == "population_rfs":
            # Pass the selected cluster_id to highlight in the population plot
            plotting.draw_population_rfs_plot(main_window, selected_cell_id=cluster_id)
        elif view_type == "timecourse":
            plotting.draw_sta_timecourse_plot(main_window, cluster_id)
        elif view_type == "animation":
            plotting.draw_sta_animation_plot(main_window, cluster_id)


def on_spatial_data_ready(main_window, cluster_id, features):
    """Callback for when heavyweight spatial features are ready from the worker."""
    current_id = main_window._get_selected_cluster_id()
    current_tab_widget = main_window.analysis_tabs.currentWidget()
    if cluster_id == current_id and current_tab_widget == main_window.summary_tab:
        plotting.draw_summary_EI_plot(main_window, cluster_id)
        main_window.status_bar.showMessage("Spatial analysis complete.", 2000)

def on_refine_cluster(main_window):
    """Starts the cluster refinement process in a background thread."""
    cluster_id = main_window._get_selected_cluster_id()
    if cluster_id is None:
        QMessageBox.warning(main_window, "No Cluster Selected", "Please select a cluster from the table to refine.")
        return
        
    main_window.refine_button.setEnabled(False)
    main_window.status_bar.showMessage(f"Starting refinement for Cluster {cluster_id}...")
    
    main_window.refine_thread = QThread()
    main_window.refinement_worker = RefinementWorker(main_window.data_manager, cluster_id)
    main_window.refinement_worker.moveToThread(main_window.refine_thread)
    main_window.refinement_worker.finished.connect(main_window.handle_refinement_results)
    main_window.refinement_worker.error.connect(main_window.handle_refinement_error)
    main_window.refinement_worker.progress.connect(lambda msg: main_window.status_bar.showMessage(msg, 3000))
    main_window.refine_thread.started.connect(main_window.refinement_worker.run)
    main_window.refine_thread.start()

def handle_refinement_results(main_window, parent_id, new_clusters):
    """Handles the results from a successful refinement operation."""
    main_window.status_bar.showMessage(f"Refinement of C{parent_id} complete. Found {len(new_clusters)} sub-clusters.", 5000)
    main_window.data_manager.update_after_refinement(parent_id, new_clusters)
    # Refresh the tree view to show updated cluster information
    populate_tree_view(main_window)
    main_window.refine_button.setEnabled(True)
    main_window.save_action.setEnabled(True)
    main_window.setWindowTitle("*axolotl (unsaved changes)")
    main_window.refine_thread.quit()
    main_window.refine_thread.wait()

def handle_refinement_error(main_window, error_message):
    """Handles an error from the refinement worker."""
    QMessageBox.critical(main_window, "Refinement Error", error_message)
    main_window.status_bar.showMessage("Refinement failed.", 5000)
    main_window.refine_button.setEnabled(True)
    main_window.refine_thread.quit()
    main_window.refine_thread.wait()

def on_save_action(main_window):
    """Handles the save action from the menu."""
    if main_window.data_manager:
        if main_window.data_manager.info_path:
            original_path = main_window.data_manager.info_path
            suggested_path = str(original_path.parent / f"{original_path.stem}_refined.tsv")
        else:
            suggested_path = str(main_window.data_manager.kilosort_dir / "cluster_group_refined.tsv")

        save_path, _ = QFileDialog.getSaveFileName(main_window, "Save Refined Cluster Info",
            suggested_path, "TSV Files (*.tsv)")
        
        if save_path:
            save_results(main_window, save_path)

def save_results(main_window, output_path):
    """Saves the refined cluster data to a TSV file."""
    try:
        col = 'KSLabel' if 'KSLabel' in main_window.data_manager.cluster_info.columns else 'group'
        final_df = main_window.data_manager.cluster_df[['cluster_id', 'KSLabel']].copy()
        final_df.rename(columns={'KSLabel': col}, inplace=True)
        final_df.to_csv(output_path, sep='\t', index=False)
        
        # Also save the tree structure to a separate JSON file
        tree_save_path = output_path.replace('.tsv', '_tree.json')
        main_window.data_manager.save_tree_structure(tree_save_path)
        
        main_window.data_manager.is_dirty = False
        main_window.setWindowTitle("axolotl")
        main_window.save_action.setEnabled(False)
        main_window.status_bar.showMessage(f"Results saved to {output_path} and tree structure to {tree_save_path}", 5000)
    except Exception as e:
        QMessageBox.critical(main_window, "Save Error", f"Could not save the file: {e}")
        main_window.status_bar.showMessage("Save failed.", 5000)

def apply_good_filter(main_window):
    """Filters the tree view to show only 'good' clusters."""
    if main_window.data_manager is None:
        return
    
    # Create a filtered tree with only 'good' clusters
    model = main_window.tree_model
    model.clear()  # Clear any previous data
    
    df = main_window.data_manager.original_cluster_df[
        main_window.data_manager.original_cluster_df['KSLabel'] == 'good'
    ].copy()

    # Update table view with filtered data
    main_window.pandas_model = PandasModel(df)
    main_window.setup_table_model(main_window.pandas_model)
    
    # Create top-level nodes for each unique KSLabel
    groups = {}
    for label in df['KSLabel'].unique():
        group_item = QStandardItem(label)
        group_item.setEditable(False)
        group_item.setDropEnabled(True)  # Can drop cells into it
        
        # Style group items differently from cells
        font = group_item.font()
        font.setBold(True)
        group_item.setFont(font)
        
        # Set different background color for groups
        group_item.setBackground(QColor('#3C3C3C'))  # Dark gray background for groups
        
        # Add folder icon for groups
        group_item.setIcon(main_window.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        
        groups[label] = group_item
        model.appendRow(group_item)
        
    # Add each cluster as a child item to its group
    for _, row in df.iterrows():
        cluster_id = row['cluster_id']
        label = row['KSLabel']
        
        # The text displayed will be e.g., "Cluster 123 (n=456 spikes)"
        item_text = f"Cluster {cluster_id} (n={row['n_spikes']})"
        cell_item = QStandardItem(item_text)
        cell_item.setEditable(False)
        
        # Add a special icon or style for cells to distinguish them
        font = cell_item.font()
        font.setItalic(False)
        cell_item.setFont(font)
        
        # Add file icon for cells
        cell_item.setIcon(main_window.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        
        # IMPORTANT: Store the actual cluster ID in the item's data role.
        # This is how we'll retrieve it when the item is clicked.
        cell_item.setData(cluster_id, Qt.ItemDataRole.UserRole)
        
        # Prevent dropping items onto cells
        cell_item.setDropEnabled(False) 
        
        groups[label].appendRow(cell_item)
        
    main_window.setup_tree_model(model)
    main_window.tree_view.expandAll()

def reset_views(main_window):
    """Resets the views to their original, unfiltered state."""
    if main_window.data_manager is None:
        return
    # Repopulate the tree with original data
    populate_tree_view(main_window)

def start_worker(main_window):
    """Starts the background spatial worker thread."""
    if main_window.worker_thread is not None:
        stop_worker(main_window)
    main_window.worker_thread = QThread()
    main_window.spatial_worker = SpatialWorker(main_window.data_manager)
    main_window.spatial_worker.moveToThread(main_window.worker_thread)
    main_window.worker_thread.started.connect(main_window.spatial_worker.run)
    main_window.spatial_worker.result_ready.connect(main_window.on_spatial_data_ready)
    main_window.worker_thread.start()

def populate_tree_view(main_window):
    """Builds the initial tree and table from the loaded Kilosort data."""
    df = main_window.data_manager.cluster_df

    # --- Populate Table View ---
    main_window.pandas_model = PandasModel(df)
    main_window.setup_table_model(main_window.pandas_model)

    # --- Populate Tree View ---
    model = main_window.tree_model
    model.clear()  # Clear any previous data
    
    df = main_window.data_manager.cluster_df
    
    # Create top-level nodes for each unique cell type
    groups = {}
    for label in df['cell_type'].unique():
        group_item = QStandardItem(label)
        group_item.setEditable(False)
        group_item.setDropEnabled(True)  # Can drop cells into it
        
        # Style group items differently from cells
        font = group_item.font()
        font.setBold(True)
        group_item.setFont(font)
        
        # Set different background color for groups
        group_item.setBackground(QColor('#3C3C3C'))  # Dark gray background for groups
        
        # Add folder icon for groups
        group_item.setIcon(main_window.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        
        groups[label] = group_item
        model.appendRow(group_item)
        
    # Add each cluster as a child item to its group
    for _, row in df.iterrows():
        cluster_id = row['cluster_id']
        label = row['cell_type']
        
        # The text displayed will be e.g., "Cluster 123 (n=456 spikes)"
        item_text = f"Cluster {cluster_id} (n={row['n_spikes']})"
        cell_item = QStandardItem(item_text)
        cell_item.setEditable(False)
        
        # Add a special icon or style for cells to distinguish them
        font = cell_item.font()
        font.setItalic(False)
        cell_item.setFont(font)
        
        # Add file icon for cells
        cell_item.setIcon(main_window.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        
        # IMPORTANT: Store the actual cluster ID in the item's data role.
        # This is how we'll retrieve it when the item is clicked.
        cell_item.setData(cluster_id, Qt.ItemDataRole.UserRole)
        
        # Prevent dropping items onto cells
        cell_item.setDropEnabled(False) 
        
        groups[label].appendRow(cell_item)
        
    main_window.setup_tree_model(model)
    main_window.tree_view.expandAll()


def add_new_group(main_window, name):
    """Adds a new top-level group to the tree view."""
    item = QStandardItem(name)
    item.setEditable(False)
    item.setDropEnabled(True)
    
    # Apply the same styling as other groups
    font = item.font()
    font.setBold(True)
    item.setFont(font)
    
    # Set different background color for groups
    item.setBackground(QColor('#3C3C3C'))  # Dark gray background for groups
    
    # Add folder icon for groups
    item.setIcon(main_window.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
    
    main_window.tree_model.appendRow(item)


def stop_worker(main_window):
    """Stops the background spatial worker thread."""
    if main_window.worker_thread and main_window.worker_thread.isRunning():
        main_window.spatial_worker.stop()
        main_window.worker_thread.quit()
        main_window.worker_thread.wait()


def load_classification_file(main_window):
    """Load a Vision classification file and populate the tree structure based on it."""
    if not main_window.data_manager:
        QMessageBox.warning(main_window, "No Data Loaded", "Please load a Kilosort directory first.")
        return

    file_path, _ = QFileDialog.getOpenFileName(
        main_window, 
        "Select Classification File", 
        str(Path.home()),
        "Text Files (*.txt);;All Files (*)"
    )
    
    if not file_path:
        return
    
    try:
        main_window.status_bar.showMessage("Loading classification file...")
        QApplication.processEvents()
        
        # Parse the classification file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Dictionary to store cluster_id -> classification path
        classifications = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                vision_id = int(parts[0])
                classification_path = parts[1]
                
                # Convert vision_id (1-indexed) to cluster_id (0-indexed)
                cluster_id = vision_id - 1
                classifications[cluster_id] = classification_path
        
        # Update the tree view based on classifications
        main_window.tree_model.clear()
        
        # Group clusters by their classifications
        classification_groups = {}
        for cluster_id, path in classifications.items():
            if path not in classification_groups:
                classification_groups[path] = []
            classification_groups[path].append(cluster_id)
        
        # Also need to maintain clusters that weren't in the classification file
        all_cluster_ids = set(main_window.data_manager.cluster_df['cluster_id'])
        classified_cluster_ids = set(classifications.keys())
        unclassified_cluster_ids = all_cluster_ids - classified_cluster_ids
        
        # Add clusters that were in the classification file
        for path, cluster_ids in classification_groups.items():
            # Create hierarchical structure from path (e.g., "All/OFF/brisk sustained/" -> nested groups)
            path_parts = [part for part in path.split('/') if part]  # Remove empty parts
            
            # Navigate/create the hierarchical structure
            current_parent = main_window.tree_model
            for i, part in enumerate(path_parts):
                # Look for existing item with this name in current parent
                found_item = None
                for row in range(current_parent.rowCount()):
                    # For QStandardItemModel (like tree_model), use item(row)
                    # For QStandardItem (nested items), use child(row)
                    if type(current_parent).__name__ == 'QStandardItemModel':
                        item = current_parent.item(row)
                    else:
                        item = current_parent.child(row)
                    if item and item.text() == part:
                        found_item = item
                        break
                
                if found_item is None:
                    # Create new group item
                    group_item = QStandardItem(part)
                    group_item.setEditable(False)
                    group_item.setDropEnabled(True)
                    
                    # Style group items differently from cells
                    font = group_item.font()
                    font.setBold(True)
                    group_item.setFont(font)
                    
                    # Set different background color for groups
                    group_item.setBackground(QColor('#3C3C3C'))  # Dark gray background for groups
                    
                    # Add folder icon for groups
                    group_item.setIcon(main_window.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
                    
                    current_parent.appendRow(group_item)
                    current_parent = group_item
                else:
                    current_parent = found_item
            
            # Add cluster items to the final group
            for cluster_id in cluster_ids:
                if cluster_id in all_cluster_ids:
                    cluster_info = main_window.data_manager.cluster_df[
                        main_window.data_manager.cluster_df['cluster_id'] == cluster_id
                    ].iloc[0]
                    
                    # Create cluster item with n_spikes and ISI info
                    item_text = f"Cluster {cluster_id} (n={cluster_info['n_spikes']}, ISI={cluster_info['isi_violations_pct']:.2f}%)"
                    cell_item = QStandardItem(item_text)
                    cell_item.setEditable(False)
                    
                    # Add file icon for cells
                    cell_item.setIcon(main_window.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
                    
                    # Store the cluster ID in the item's data role
                    cell_item.setData(cluster_id, Qt.ItemDataRole.UserRole)
                    
                    # Prevent dropping items onto cells
                    cell_item.setDropEnabled(False)
                    
                    current_parent.appendRow(cell_item)
        
        # Add unclassified clusters under an 'Unclassified' group
        if unclassified_cluster_ids:
            unclassified_group = QStandardItem("Unclassified")
            unclassified_group.setEditable(False)
            unclassified_group.setDropEnabled(True)
            
            # Style group items differently from cells
            font = unclassified_group.font()
            font.setBold(True)
            unclassified_group.setFont(font)
            
            # Set different background color for groups
            unclassified_group.setBackground(QColor('#3C3C3C'))  # Dark gray background for groups
            
            # Add folder icon for groups
            unclassified_group.setIcon(main_window.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
            
            main_window.tree_model.appendRow(unclassified_group)
            
            for cluster_id in unclassified_cluster_ids:
                cluster_info = main_window.data_manager.cluster_df[
                    main_window.data_manager.cluster_df['cluster_id'] == cluster_id
                ].iloc[0]
                
                # Create cluster item with n_spikes and ISI info
                item_text = f"Cluster {cluster_id} (n={cluster_info['n_spikes']}, ISI={cluster_info['isi_violations_pct']:.2f}%)"
                cell_item = QStandardItem(item_text)
                cell_item.setEditable(False)
                
                # Add file icon for cells
                cell_item.setIcon(main_window.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
                
                # Store the cluster ID in the item's data role
                cell_item.setData(cluster_id, Qt.ItemDataRole.UserRole)
                
                # Prevent dropping items onto cells
                cell_item.setDropEnabled(False)
                
                unclassified_group.appendRow(cell_item)
        
        # Set up the tree model and expand all
        main_window.setup_tree_model(main_window.tree_model)
        main_window.tree_view.expandAll()
        
        main_window.status_bar.showMessage(f"Loaded classification file with {len(classifications)} classified clusters.", 5000)
        main_window.load_classification_action.setEnabled(True)
        
    except Exception as e:
        QMessageBox.critical(main_window, "Loading Error", f"Error loading classification file: {e}")
        main_window.status_bar.showMessage("Classification file loading failed.", 5000)


def on_go_to_time(main_window):
    """
    Handle the 'Go' button click to navigate to a specific time in the raw trace view.
    """
    if main_window.data_manager.raw_data_memmap is None:
        main_window.status_bar.showMessage("No raw data loaded.", 5000)
        return

    # Get time values from the separate input boxes
    hours_text = main_window.time_input_hours.text().strip()
    minutes_text = main_window.time_input_minutes.text().strip()
    seconds_text = main_window.time_input_seconds.text().strip()
    
    if not hours_text or not minutes_text or not seconds_text:
        main_window.status_bar.showMessage("Please enter values for hours, minutes, and seconds.", 5000)
        return

    try:
        # Parse the time components
        hours = int(hours_text)
        minutes = int(minutes_text)
        seconds = int(seconds_text)
        
        # Validate ranges
        if hours < 0 or hours > 23:
            raise ValueError("Hours must be between 0 and 23")
        if minutes < 0 or minutes > 59:
            raise ValueError("Minutes must be between 0 and 59")
        if seconds < 0 or seconds > 59:
            raise ValueError("Seconds must be between 0 and 59")
        
        total_seconds = hours * 3600 + minutes * 60 + seconds
        
        # Calculate the sample corresponding to the requested time
        target_sample = int(total_seconds * main_window.data_manager.sampling_rate)
        
        if target_sample >= main_window.data_manager.n_samples:
            main_window.status_bar.showMessage(f"Time {hours:02d}:{minutes:02d}:{seconds:02d} exceeds recording duration.", 5000)
            return
        
        # Show a fixed window starting from the target time (e.g., 5 seconds from target)
        duration_to_show = 5.0  # seconds
        
        start_sample = target_sample
        end_sample = min(main_window.data_manager.n_samples, target_sample + int(duration_to_show * main_window.data_manager.sampling_rate))
        
        # Convert samples back to seconds for the plot
        start_time = start_sample / main_window.data_manager.sampling_rate
        end_time = end_sample / main_window.data_manager.sampling_rate
        
        # Set the X-axis range of the raw trace plot
        main_window.raw_trace_plot.setXRange(start_time, end_time, padding=0)
        
        # Update the plot for the currently selected cluster
        cluster_id = main_window._get_selected_cluster_id()
        if cluster_id is not None:
            main_window.load_raw_trace_data(cluster_id)
        
        main_window.status_bar.showMessage(f"Jumped to time {hours:02d}:{minutes:02d}:{seconds:02d}", 2000)
        
    except ValueError:
        main_window.status_bar.showMessage("Invalid time values. Use valid numbers (HH:MM:SS, e.g., 00:05:30).", 5000)