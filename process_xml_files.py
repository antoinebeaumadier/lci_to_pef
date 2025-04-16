import os
import pandas as pd
import xml.etree.ElementTree as ET
from format_xml import process_xml_directory
import numpy as np
from functools import lru_cache
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
from collections import defaultdict

# Suppress pandas performance warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class FlowCache:
    """Cache for flow names from UUIDs"""
    def __init__(self, flows_dir='flows'):
        self.flows_dir = flows_dir
        self._cache = {}  # uuid -> {name, basename, synonyms}
    
    @lru_cache(maxsize=1000)
    def get_flow_info(self, uuid):
        """Get flow information (name, basename, synonyms) from UUID using cache"""
        if uuid in self._cache:
            return self._cache[uuid]
        
        # Try to find the flow file
        flow_file = os.path.join(self.flows_dir, f"{uuid}.xml")
        if not os.path.exists(flow_file):
            return None
        
        try:
            # Parse the XML file
            tree = ET.parse(flow_file)
            root = tree.getroot()
            
            # Define namespaces
            ns = {
                'common': 'http://www.EcoInvent.org/EcoSpold01',
                'ns3': 'http://www.EcoInvent.org/Extensions'
            }
            
            # Get flow information
            flow_info = {}
            
            # Get short description (main name)
            short_desc = root.find('.//common:shortDescription', ns)
            if short_desc is not None:
                flow_info['name'] = short_desc.text.strip()
            
            # Get baseName if available
            base_name = root.find('.//ns3:baseName', ns)
            if base_name is not None:
                flow_info['basename'] = base_name.text.strip()
            
            # Get synonyms if available
            synonyms = []
            for synonym in root.findall('.//ns3:synonym', ns):
                if synonym.text:
                    synonyms.append(synonym.text.strip())
            if synonyms:
                flow_info['synonyms'] = synonyms
            
            if flow_info:
                self._cache[uuid] = flow_info
                return flow_info
                
        except Exception as e:
            print(f"Error parsing flow file {flow_file}: {str(e)}")
        
        return None
    
    def get_flow_name(self, uuid):
        """Get primary flow name from UUID"""
        info = self.get_flow_info(uuid)
        return info['name'] if info and 'name' in info else None

# Initialize global flow cache
flow_cache = FlowCache()

# Cache for normalized values
normalized_cache = {}

@lru_cache(maxsize=10000)
def cached_normalize_flow_name(name):
    """Cached version of normalize_flow_name"""
    if name is None:
        return None
    if not isinstance(name, str):
        return str(name)
    
    # Remove content in brackets and parentheses
    name = re.sub(r'\[.*?\]', '', name)  # Remove everything in square brackets
    name = re.sub(r'\(.*?\)', '', name)  # Remove everything in parentheses
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove special characters but keep spaces, hyphens, and commas
    name = re.sub(r'[^\w\s,-]', '', name)
    
    # Replace multiple spaces with single space
    name = ' '.join(name.split())
    
    # Remove common prefixes and suffixes
    name = re.sub(r'^(the|a|an)\s+', '', name)
    name = re.sub(r'\s+(the|a|an)$', '', name)
    
    # Handle chemical compound names
    # Remove common chemical prefixes
    name = re.sub(r'^(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)-', '', name)
    
    # Remove common chemical suffixes
    name = re.sub(r'-(ic|ous|ide|ate|ite|ol|al|one|ene|yne|ane)$', '', name)
    
    # Remove CAS numbers and other identifiers
    name = re.sub(r'\d{2,7}-\d{2}-\d', '', name)  # CAS numbers
    name = re.sub(r'[A-Z]{2,3}\d{4,6}', '', name)  # Common chemical identifiers
    
    # Handle chemical compound variations
    # Remove spaces around hyphens in chemical names
    name = re.sub(r'\s*-\s*', '-', name)
    
    # Remove spaces around commas in chemical names
    name = re.sub(r'\s*,\s*', ',', name)
    
    # Standardize common chemical prefixes
    name = re.sub(r'^1,1,1-', '1,1,1-', name)
    name = re.sub(r'^1,1-', '1,1-', name)
    name = re.sub(r'^1,2-', '1,2-', name)
    name = re.sub(r'^1,3-', '1,3-', name)
    name = re.sub(r'^1,4-', '1,4-', name)
    name = re.sub(r'^2,3-', '2,3-', name)
    name = re.sub(r'^2,4-', '2,4-', name)
    
    # Standardize common chemical suffixes
    name = re.sub(r'-dichloro$', '-dichloride', name)
    name = re.sub(r'-trichloro$', '-trichloride', name)
    name = re.sub(r'-tetrachloro$', '-tetrachloride', name)
    name = re.sub(r'-pentachloro$', '-pentachloride', name)
    name = re.sub(r'-hexachloro$', '-hexachloride', name)
    
    # Standardize common chemical names
    name = re.sub(r'ethylene', 'ethene', name)
    name = re.sub(r'propylene', 'propene', name)
    name = re.sub(r'butylene', 'butene', name)
    name = re.sub(r'benzene', 'benzol', name)
    name = re.sub(r'toluene', 'methylbenzene', name)
    name = re.sub(r'xylene', 'dimethylbenzene', name)
    
    # Remove any remaining special characters
    name = re.sub(r'[^\w\s,-]', '', name)
    
    # Final cleanup
    return name.strip()

@lru_cache(maxsize=1000)
def cached_normalize_compartment(compartment):
    """Cached version of normalize_compartment"""
    if pd.isna(compartment):
        return 'emissions to air'  # Default to emissions to air if not specified
    
    # Convert to lowercase and strip whitespace
    compartment = str(compartment).lower().strip()
    
    # Map common variations to standard names
    compartment_map = {
        'air': 'emissions to air',
        'soil': 'emissions to soil',
        'water': 'emissions to water',
        'resource': 'resource flow',
        'unspecified': 'emissions to air',  # Default unspecified to air
        'long-term': 'emissions to air',    # Default long-term to air
        'lower stratosphere': 'emissions to air',
        'upper troposphere': 'emissions to air',
        'sea water': 'emissions to water',
        'fresh water': 'emissions to water',
        'ground water': 'emissions to water',
        'surface water': 'emissions to water'
    }
    
    # Check if the compartment contains any of the mapped keywords
    for key, value in compartment_map.items():
        if key in compartment:
            return value
    
    # If no match found, try to extract the main compartment type
    if 'air' in compartment:
        return 'emissions to air'
    elif 'water' in compartment:
        return 'emissions to water'
    elif 'soil' in compartment:
        return 'emissions to soil'
    elif 'resource' in compartment:
        return 'resource flow'
    
    return compartment

def extract_uuid(flow_name):
    """Extract UUID from a flow name that may contain brackets or parentheses"""
    if not flow_name:
        return None
        
    # Try different formats:
    # 1. Before square brackets: "uuid [compartment]"
    # 2. Before parentheses: "uuid (compartment)"
    # 3. Plain UUID format
    parts = flow_name.split('[')[0].split('(')[0].strip()
    
    # Check if the remaining part matches UUID format (8-4-4-4-12 hexadecimal)
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    if uuid_pattern.match(parts):
        return parts
    return None

def find_matching_flow(flow_name, compartment, cf_matrix):
    """Find the matching flow name in the characterization matrix"""
    # First try to find matches using UUID and flow info
    try:
        uuid = extract_uuid(flow_name)
        if uuid:
            flow_info = flow_cache.get_flow_info(uuid)
            if flow_info:
                # Get normalized values from cache or compute them
                normalized_matrix_names = cf_matrix.index.get_level_values('Flow name').map(cached_normalize_flow_name)
                normalized_compartment = cached_normalize_compartment(compartment)
                
                # Try matching with all available names
                for name_type, name_value in flow_info.items():
                    if name_type == 'synonyms':
                        # Try each synonym
                        for synonym in name_value:
                            # Try exact match with compartment
                            if (synonym, compartment) in cf_matrix.index:
                                return [(synonym, compartment)]
                            
                            # Try normalized match with compartment
                            normalized_name = cached_normalize_flow_name(synonym)
                            if normalized_name:
                                # Try exact compartment match first
                                matches = (normalized_matrix_names == normalized_name) & (cf_matrix.index.get_level_values('Compartment') == normalized_compartment)
                                if matches.any():
                                    return [idx for idx in cf_matrix.index[matches]]
                                
                                # If no exact compartment match, try any compartment
                                matches = normalized_matrix_names == normalized_name
                                if matches.any():
                                    return [idx for idx in cf_matrix.index[matches]]
                    else:
                        # Try exact match with compartment for name or basename
                        if (name_value, compartment) in cf_matrix.index:
                            return [(name_value, compartment)]
                        
                        # Try normalized match with compartment
                        normalized_name = cached_normalize_flow_name(name_value)
                        if normalized_name:
                            # Try exact compartment match first
                            matches = (normalized_matrix_names == normalized_name) & (cf_matrix.index.get_level_values('Compartment') == normalized_compartment)
                            if matches.any():
                                return [idx for idx in cf_matrix.index[matches]]
                            
                            # If no exact compartment match, try any compartment
                            matches = normalized_matrix_names == normalized_name
                            if matches.any():
                                return [idx for idx in cf_matrix.index[matches]]
    except Exception as e:
        print(f"Error trying to find exact flow name match: {str(e)}")
    
    # If no match found with UUID info, try regular matching
    # Normalize the flow name
    normalized_flow = cached_normalize_flow_name(flow_name)
    if normalized_flow is None:
        return None
    
    # Try exact match with compartment
    if (flow_name, compartment) in cf_matrix.index:
        return [(flow_name, compartment)]
    
    # Try normalized match with compartment
    normalized_matrix_names = cf_matrix.index.get_level_values('Flow name').map(cached_normalize_flow_name)
    normalized_compartment = cached_normalize_compartment(compartment)
    
    # Try exact compartment match first
    matches = (normalized_matrix_names == normalized_flow) & (cf_matrix.index.get_level_values('Compartment') == normalized_compartment)
    if matches.any():
        return [idx for idx in cf_matrix.index[matches]]
    
    # Try matching with original flow names
    if 'Flow name original' in cf_matrix.columns:
        original_matches = (cf_matrix['Flow name original'].map(cached_normalize_flow_name) == normalized_flow) & (cf_matrix.index.get_level_values('Compartment') == normalized_compartment)
        if original_matches.any():
            return [idx for idx in cf_matrix.index[original_matches]]
    
    # Try matching with substance names from statistics
    if 'Substance name as commonly in the statistics' in cf_matrix.columns:
        stats_matches = (cf_matrix['Substance name as commonly in the statistics'].map(cached_normalize_flow_name) == normalized_flow) & (cf_matrix.index.get_level_values('Compartment') == normalized_compartment)
        if stats_matches.any():
            return [idx for idx in cf_matrix.index[stats_matches]]
    
    # If no match found with compartment, try without compartment
    matches = normalized_matrix_names == normalized_flow
    if matches.any():
        return [idx for idx in cf_matrix.index[matches]]
    
    # Try one last time with original flow names without compartment
    if 'Flow name original' in cf_matrix.columns:
        original_matches = cf_matrix['Flow name original'].map(cached_normalize_flow_name) == normalized_flow
        if original_matches.any():
            return [idx for idx in cf_matrix.index[original_matches]]
    
    # Try one last time with substance names from statistics without compartment
    if 'Substance name as commonly in the statistics' in cf_matrix.columns:
        stats_matches = cf_matrix['Substance name as commonly in the statistics'].map(cached_normalize_flow_name) == normalized_flow
        if stats_matches.any():
            return [idx for idx in cf_matrix.index[stats_matches]]
    
    # If no match found, return None to indicate unmatched flow
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
        'resource': 'resource flow',
        'unspecified': 'emissions to air',  # Default unspecified to air
        'long-term': 'emissions to air',    # Default long-term to air
        'lower stratosphere': 'emissions to air',
        'upper troposphere': 'emissions to air',
        'sea water': 'emissions to water',
        'fresh water': 'emissions to water',
        'ground water': 'emissions to water',
        'surface water': 'emissions to water'
    }
    
    # Check if the compartment contains any of the mapped keywords
    for key, value in compartment_map.items():
        if key in compartment:
            return value
    
    # If no match found, try to extract the main compartment type
    if 'air' in compartment:
        return 'emissions to air'
    elif 'water' in compartment:
        return 'emissions to water'
    elif 'soil' in compartment:
        return 'emissions to soil'
    elif 'resource' in compartment:
        return 'resource flow'
    
    return compartment

def calculate_process_impacts(process_name, lci_data, cf_matrix, impact_categories):
    """Calculate impacts for a single process"""
    # Get flows for this process
    process_flows = lci_data[lci_data['Process name'] == process_name]
    
    # Initialize process impacts array
    process_impacts = np.zeros(len(impact_categories))
    
    # Track matched and unmatched flows
    matched_count = 0
    unmatched_flows = []
    
    # Calculate impacts for each flow
    for _, flow in process_flows.iterrows():
        flow_name = flow['Flow name']
        amount = flow['Amount']
        compartment = flow['Compartment']
        
        # Skip invalid amounts or names
        if pd.isna(amount) or pd.isna(flow_name) or not isinstance(flow_name, str) or not flow_name.strip():
            continue
        
        try:
            # Convert amount to float
            amount = float(amount)
            
            # Find matching flow(s)
            matches = find_matching_flow(flow_name, compartment, cf_matrix)
            if matches is not None and len(matches) > 0:
                matched_count += 1
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
                # Add to unmatched flows list
                unmatched_flows.append({
                    'Process name': process_name,
                    'Flow name': flow_name,
                    'Compartment': compartment,
                    'Amount': amount,
                    'Normalized flow name': cached_normalize_flow_name(flow_name),
                    'Normalized compartment': cached_normalize_compartment(compartment)
                })
        except Exception as e:
            print(f"Error processing flow '{flow_name}' in process '{process_name}': {str(e)}")
            continue
    
    return process_name, process_impacts, matched_count, unmatched_flows

def calculate_impacts(lci_data, cf_matrix):
    """
    Calculate environmental impacts for each file based on LCI data and characterization factors.
    
    Args:
        lci_data (pd.DataFrame): DataFrame containing LCI data with columns: Flow name, Amount, Process name
        cf_matrix (pd.DataFrame): DataFrame containing characterization factors indexed by flow names
        
    Returns:
        pd.DataFrame: DataFrame with impact results per process
    """
    # Get impact categories
    impact_categories = [col for col in cf_matrix.columns if col != 'Flow name original']
    
    # Initialize results DataFrame with zeros
    unique_processes = lci_data['Process name'].unique()
    results = pd.DataFrame(0.0, index=unique_processes, columns=impact_categories)
    
    # Initialize counters for detailed statistics
    total_flows = len(lci_data)
    total_matched_flows = 0
    invalid_amounts = sum(lci_data['Amount'].isna())
    invalid_names = sum(lci_data['Flow name'].isna() | ~lci_data['Flow name'].astype(str).str.strip().astype(bool))
    
    # Initialize list to store unmatched flows
    all_unmatched_flows = []
    
    # Convert characterization factors to numeric, replacing non-numeric values with 0
    print("\nConverting characterization factors to numeric values...")
    for col in tqdm(impact_categories, desc="Converting factors", position=0, leave=True):
        cf_matrix[col] = pd.to_numeric(cf_matrix[col], errors='coerce').fillna(0)
    
    # Determine number of workers (use 75% of available CPUs)
    num_workers = max(1, int(cpu_count() * 0.75))
    print(f"\nUsing {num_workers} workers for parallel processing")
    
    # Create a partial function with the required parameters
    process_func = partial(calculate_process_impacts, lci_data=lci_data, cf_matrix=cf_matrix, impact_categories=impact_categories)
    
    # Process in parallel with progress bar
    print("\nCalculating impacts for each process...")
    
    # Calculate optimal chunksize
    chunksize = max(1, len(unique_processes) // (num_workers * 4))
    
    with Pool(num_workers) as pool:
        # Use imap_unordered for better parallelization
        process_results = []
        for result in tqdm(
            pool.imap_unordered(process_func, unique_processes, chunksize=chunksize),
            total=len(unique_processes),
            desc="Processing processes",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ):
            process_name, impacts, matched_count, unmatched_flows = result
            process_results.append((process_name, impacts))
            total_matched_flows += matched_count
            all_unmatched_flows.extend(unmatched_flows)
    
    print("\nUpdating results...")
    # Update results with parallel processing results
    for process_name, impacts in tqdm(process_results, desc="Updating results", position=0, leave=True):
        results.loc[process_name] = impacts
    
    # Save unmatched flows to CSV
    if all_unmatched_flows:
        unmatched_df = pd.DataFrame(all_unmatched_flows)
        unmatched_df.to_csv('unmatched_flows.csv', index=False)
        print(f"\nSaved {len(all_unmatched_flows)} unmatched flows to unmatched_flows.csv")
    
    # Report detailed matching statistics
    print(f"\nDetailed Flow Processing Statistics:")
    print(f"Total flows processed: {total_flows}")
    print(f"Matched flows: {total_matched_flows} ({total_matched_flows/total_flows*100:.1f}%)")
    print(f"Unmatched flows: {len(all_unmatched_flows)} ({len(all_unmatched_flows)/total_flows*100:.1f}%)")
    print(f"Flows with invalid amounts: {invalid_amounts} ({invalid_amounts/total_flows*100:.1f}%)")
    print(f"Flows with invalid names: {invalid_names} ({invalid_names/total_flows*100:.1f}%)")
    
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
        cf_matrix['Compartment'] = cf_matrix['Compartment'].map(cached_normalize_compartment)
        
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
        lci_data['Compartment'] = lci_data['Compartment'].map(cached_normalize_compartment)
        
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