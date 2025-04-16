import os
import pandas as pd
import xml.etree.ElementTree as ET
from format_xml import process_xml_directory
import numpy as np

def load_characterization_matrix(file_path):
    """Load the characterization matrix from a CSV file."""
    # Read the CSV file, skipping the first row which is the title
    cf_matrix = pd.read_csv(file_path, sep=';', skiprows=1)
    
    # Get the impact categories (columns 10-25)
    impact_categories = cf_matrix.columns[10:26].tolist()
    
    # Create a multi-index using Flow name and Compartment
    cf_matrix.set_index(['Flow name', 'Compartment'], inplace=True)
    
    # Keep only the impact categories and original flow name
    cf_matrix = cf_matrix[impact_categories + ['Flow name original']]
    
    return cf_matrix

def normalize_flow_name(name):
    """Normalize a flow name for matching purposes"""
    if name is None:
        return None
    if not isinstance(name, str):
        return str(name)
    
    # Remove content in brackets and parentheses
    name = name.split('[')[0].split('(')[0]
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove special characters and multiple spaces
    name = ''.join(c for c in name if c.isalnum() or c.isspace())
    name = ' '.join(name.split())
    
    return name.strip()

def find_matching_flow(flow_name, compartment, cf_matrix):
    """Find the matching flow name in the characterization matrix"""
    # Normalize the flow name
    normalized_flow = normalize_flow_name(flow_name)
    if normalized_flow is None:
        return None
    
    # Try exact match with compartment
    if (flow_name, compartment) in cf_matrix.index:
        return [(flow_name, compartment)]
    
    # Try normalized match with compartment
    normalized_matrix_names = cf_matrix.index.get_level_values('Flow name').map(normalize_flow_name)
    matches = (normalized_matrix_names == normalized_flow) & (cf_matrix.index.get_level_values('Compartment') == compartment)
    if matches.any():
        return [idx for idx in cf_matrix.index[matches]]
    
    # Try matching with original flow names
    if 'Flow name original' in cf_matrix.columns:
        original_matches = (cf_matrix['Flow name original'].map(normalize_flow_name) == normalized_flow) & (cf_matrix.index.get_level_values('Compartment') == compartment)
        if original_matches.any():
            return [idx for idx in cf_matrix.index[original_matches]]
    
    # If no match found with compartment, try without compartment
    matches = normalized_matrix_names == normalized_flow
    if matches.any():
        return [idx for idx in cf_matrix.index[matches]]
    
    # Try one last time with original flow names without compartment
    if 'Flow name original' in cf_matrix.columns:
        original_matches = cf_matrix['Flow name original'].map(normalize_flow_name) == normalized_flow
        if original_matches.any():
            return [idx for idx in cf_matrix.index[original_matches]]
    
    return None

def normalize_compartment(compartment):
    """Normalize compartment names for matching purposes"""
    if pd.isna(compartment):
        return 'emissions to air'  # Default to emissions to air if not specified
    
    # Convert to lowercase and strip whitespace
    compartment = str(compartment).lower().strip()
    
    # Map common variations to standard names
    compartment_map = {
        'air': 'emissions to air',
        'soil': 'emissions to soil',
        'water': 'emissions to water',
        'resource': 'resource flow'
    }
    
    # Check if the compartment contains any of the mapped keywords
    for key, value in compartment_map.items():
        if key in compartment:
            return value
    
    return compartment

def calculate_impacts(lci_data, cf_matrix):
    """
    Calculate environmental impacts for each file based on LCI data and characterization factors.
    
    Args:
        lci_data (pd.DataFrame): DataFrame containing LCI data with columns: Flow name, Amount, Process name
        cf_matrix (pd.DataFrame): DataFrame containing characterization factors indexed by flow names
        
    Returns:
        pd.DataFrame: DataFrame with impact results per process
    """
    # Get impact categories (all columns except Flow name original)
    impact_categories = [col for col in cf_matrix.columns if col != 'Flow name original']
    
    # Initialize results DataFrame with zeros
    unique_processes = lci_data['Process name'].unique()
    results = pd.DataFrame(0.0, index=unique_processes, columns=impact_categories)
    
    # Initialize counters for statistics
    total_flows = 0
    matched_flows = 0
    unmatched_flows = set()
    
    # Convert characterization factors to numeric, replacing non-numeric values with 0
    for col in impact_categories:
        cf_matrix[col] = pd.to_numeric(cf_matrix[col], errors='coerce').fillna(0)
    
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
            compartment = flow['Compartment']
            
            # Skip if amount is not valid
            if pd.isna(amount):
                print(f"Warning: Invalid amount for flow '{flow_name}' in process '{process_name}'")
                continue
            
            try:
                # Convert amount to float
                amount = float(amount)
                
                # Find matching flow(s)
                matches = find_matching_flow(flow_name, compartment, cf_matrix)
                if matches is not None:
                    matched_flows += 1
                    # Sum up characterization factors from all matches
                    total_factors = np.zeros(len(impact_categories))
                    for match in matches:
                        try:
                            factors = cf_matrix.loc[match, impact_categories].values
                            if isinstance(factors, pd.Series):
                                factors = factors.values
                            factors = factors.astype(float)  # Ensure numeric type
                            if len(factors.shape) > 1:
                                factors = factors.mean(axis=0)  # Take mean if multiple values
                            total_factors += factors
                        except Exception as e:
                            print(f"Warning: Error processing factors for flow '{flow_name}' match {match}: {str(e)}")
                            continue
                    
                    # If we found multiple matches, take the average
                    if len(matches) > 1:
                        total_factors /= len(matches)
                    
                    # Multiply amount by characterization factors and add to process impacts
                    impact = total_factors * amount
                    
                    # Check for invalid values before adding
                    if not np.any(np.isnan(impact)) and not np.any(np.isinf(impact)):
                        process_impacts += impact
                    else:
                        print(f"Warning: Invalid impact values for flow '{flow_name}' in process '{process_name}'")
                        print(f"Amount: {amount}")
                        print(f"Factors: {total_factors}")
                        print(f"Impact: {impact}")
                else:
                    unmatched_flows.add(f"{flow_name} ({compartment})")
            except Exception as e:
                print(f"Error processing flow '{flow_name}' in process '{process_name}': {str(e)}")
                continue
        
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
        # Read the CSV file, skipping the first row which contains the title
        cf_matrix = pd.read_csv('cf_matrix.csv', sep=';', skiprows=1)
        
        # Get the impact categories (columns 10-25, which are the characterization factors)
        impact_categories = cf_matrix.columns[10:26].tolist()
        
        # Clean flow names and normalize compartments
        cf_matrix['Flow name'] = cf_matrix['Flow name'].str.strip()
        cf_matrix['Compartment'] = cf_matrix['Compartment'].map(normalize_compartment)
        
        # Create a multi-index using Flow name and Compartment
        cf_matrix.set_index(['Flow name', 'Compartment'], inplace=True)
        
        # Keep only the impact categories and original flow name
        cf_matrix = cf_matrix[impact_categories + ['Flow name original']]
        
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
        
        # Clean flow names and normalize compartments in LCI data
        lci_data['Flow name'] = lci_data['Flow name'].str.strip()
        lci_data['Compartment'] = lci_data['Compartment'].map(normalize_compartment)
        
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