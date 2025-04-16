import os
import pandas as pd
import xml.etree.ElementTree as ET
from format_xml import process_xml_directory
import numpy as np

def load_characterization_matrix(file_path):
    """Load the characterization matrix from a CSV file."""
    cf_matrix = pd.read_csv(file_path, sep=';')
    cf_matrix.set_index('Flow name', inplace=True)
    return cf_matrix

def normalize_flow_name(name):
    """Normalize a flow name for matching purposes"""
    if name is None:
        return None
    if not isinstance(name, str):
        return str(name)
    # Convert to lowercase
    name = name.lower()
    # Remove special characters and multiple spaces
    name = ''.join(c for c in name if c.isalnum() or c.isspace())
    name = ' '.join(name.split())
    return name

def find_matching_flow(flow_name, cf_matrix):
    """Find the matching flow name in the characterization matrix"""
    # Normalize the flow name
    normalized_flow = normalize_flow_name(flow_name)
    if normalized_flow is None:
        return None
    
    # Check for exact match first
    if flow_name in cf_matrix.index:
        return flow_name
    
    # Check for normalized match
    normalized_matrix_names = cf_matrix.index.map(normalize_flow_name)
    matches = normalized_matrix_names == normalized_flow
    if matches.any():
        return cf_matrix.index[matches].iloc[0]
    
    # If 'Flow name original' column exists, check it
    if 'Flow name original' in cf_matrix.columns:
        if flow_name in cf_matrix['Flow name original'].values:
            mask = cf_matrix['Flow name original'] == flow_name
            return cf_matrix.loc[mask, 'Flow name'].iloc[0]
        
        # Check normalized original names
        normalized_original_names = cf_matrix['Flow name original'].map(normalize_flow_name)
        matches = normalized_original_names == normalized_flow
        if matches.any():
            return cf_matrix.loc[matches, 'Flow name'].iloc[0]
    
    return None

def calculate_impacts(lci_data, cf_matrix):
    """
    Calculate environmental impacts for each file based on LCI data and characterization factors.
    
    Args:
        lci_data (pd.DataFrame): DataFrame containing LCI data with columns: Flow name, Amount, Process name
        cf_matrix (pd.DataFrame): DataFrame containing characterization factors indexed by flow names
        
    Returns:
        pd.DataFrame: DataFrame with impact results per process
    """
    # Get impact categories (all columns except index)
    impact_categories = cf_matrix.columns.tolist()
    
    # Initialize results DataFrame with zeros
    unique_processes = lci_data['Process name'].unique()
    results = pd.DataFrame(0.0, index=unique_processes, columns=impact_categories)
    
    # Initialize counters for statistics
    total_flows = 0
    matched_flows = 0
    unmatched_flows = set()
    
    # Process each process separately
    for process_name in unique_processes:
        # Get flows for this process
        process_flows = lci_data[lci_data['Process name'] == process_name]
        total_flows += len(process_flows)
        
        # Initialize process impacts array
        process_impacts = np.zeros(len(impact_categories))
        
        # Calculate impacts for each flow
        for _, flow in process_flows.iterrows():
            flow_name = flow['Flow name']
            amount = flow['Amount']
            
            # Skip if amount is not valid
            if pd.isna(amount):
                continue
                
            # Check if flow exists in characterization matrix
            if flow_name in cf_matrix.index:
                matched_flows += 1
                # Get characterization factors for all impact categories
                factors = cf_matrix.loc[flow_name].to_numpy()  # Convert to numpy array
                # Multiply amount by characterization factors and add to process impacts
                impact = factors * float(amount)
                process_impacts = process_impacts + impact
            else:
                unmatched_flows.add(flow_name)
        
        # Assign the total impacts for this process
        results.loc[process_name] = process_impacts
    
    # Report matching statistics
    print(f"\nFlow matching statistics:")
    print(f"Total flows processed: {total_flows}")
    if total_flows > 0:
        print(f"Matched flows: {matched_flows} ({matched_flows/total_flows*100:.1f}%)")
        print(f"Unmatched flows: {len(unmatched_flows)} ({len(unmatched_flows)/total_flows*100:.1f}%)")
    
    # Report unmatched flows
    if unmatched_flows:
        print("\nWarning: The following flows were not found in the characterization matrix (showing first 20):")
        for flow in sorted(list(unmatched_flows))[:20]:
            print(f"  - {flow}")
        if len(unmatched_flows) > 20:
            print(f"  ... and {len(unmatched_flows) - 20} more")
    
    return results

def main():
    try:
        # Load characterization matrix
        print("Loading characterization matrix...")
        # Skip the first row which contains the title
        cf_matrix = pd.read_csv('cf_matrix.csv', sep=';', skiprows=1)
        
        # Get the impact categories (columns 10-25, which are the characterization factors)
        impact_categories = cf_matrix.columns[10:26].tolist()
        
        # Clean flow names and set as index
        cf_matrix['Flow name'] = cf_matrix['Flow name'].str.strip()
        
        # Convert numeric columns to float, replacing empty strings with NaN
        for col in impact_categories:
            cf_matrix[col] = pd.to_numeric(cf_matrix[col], errors='coerce').fillna(0)
        
        # Handle duplicate flow names by summing their characterization factors
        cf_matrix = cf_matrix.groupby('Flow name')[impact_categories].sum()
        
        # Debug: Print characterization matrix info
        print("\nCharacterization matrix info:")
        print(f"Shape: {cf_matrix.shape}")
        print(f"Index type: {type(cf_matrix.index)}")
        print(f"Column types:\n{cf_matrix.dtypes}")
        print("\nFirst few rows of characterization matrix:")
        print(cf_matrix.head())
        
        # Process XML files
        print("\nProcessing XML files...")
        lci_data = process_xml_directory('.')
        if lci_data.empty:
            print("No LCI data found to process")
            return
        
        # Clean flow names in LCI data
        lci_data['Flow name'] = lci_data['Flow name'].str.strip()
        
        # Debug: Print LCI data info
        print("\nLCI data info:")
        print(f"Shape: {lci_data.shape}")
        print(f"Columns: {lci_data.columns.tolist()}")
        print(f"Flow name type: {lci_data['Flow name'].dtype}")
        print(f"Amount type: {lci_data['Amount'].dtype}")
        print("\nFirst few rows of LCI data:")
        print(lci_data.head())
            
        print(f"\nFound {len(lci_data)} exchanges across {len(lci_data['Process name'].unique())} processes")
        
        # Calculate impacts
        print("\nCalculating environmental impacts...")
        results = calculate_impacts(lci_data, cf_matrix)
        
        # Save results
        print("\nSaving results...")
        results.to_excel('impacts_results.xlsx')
        print("Results saved to impacts_results.xlsx")
        
        # Print summary statistics
        print("\nSummary of results:")
        print(f"Number of processes processed: {len(results)}")
        print("\nMean impacts across all processes:")
        print(results.mean())
        
        # Save detailed results with process names
        detailed_results = results.copy()
        detailed_results.to_excel('detailed_impacts.xlsx')
        print("\nDetailed results saved to detailed_impacts.xlsx")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 