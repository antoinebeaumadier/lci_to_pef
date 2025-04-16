import os
import pandas as pd
import xml.etree.ElementTree as ET

def extract_exchanges(xml_file):
    """
    Extract exchanges from an XML file, including flow names and amounts.
    Returns a list of dictionaries containing exchange information.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Define namespace mapping
        ns = {
            'default': 'http://lca.jrc.it/ILCD/Process',
            'common': 'http://lca.jrc.it/ILCD/Common',
            'xml': 'http://www.w3.org/XML/1998/namespace'
        }
        
        # Extract baseName from the XML - get English version
        name_element = root.find('.//default:name', ns)
        base_name = None
        if name_element is not None:
            for base_name_elem in name_element.findall('default:baseName', ns):
                if base_name_elem.get('{http://www.w3.org/XML/1998/namespace}lang') == 'en':
                    base_name = base_name_elem.text.strip()
                    break
        
        # Fallback to filename if no English baseName found
        if base_name is None:
            base_name = os.path.basename(xml_file)
        
        exchanges = []
        
        # Find all exchange elements
        for exchange in root.findall('.//default:exchanges/default:exchange', ns):
            try:
                # Get flow data set reference
                flow_ref = exchange.find('.//default:referenceToFlowDataSet', ns)
                if flow_ref is None:
                    continue
                
                # Extract flow name from shortDescription
                short_desc = flow_ref.find('.//common:shortDescription', ns)
                if short_desc is None:
                    continue
                
                flow_name = short_desc.text.strip()
                if not flow_name:
                    continue
                
                # Get exchange direction (input/output)
                direction = exchange.find('default:exchangeDirection', ns)
                if direction is None:
                    continue
                
                is_input = direction.text.strip() == 'Input'
                
                # Get mean amount
                mean_amount = exchange.find('default:meanAmount', ns)
                if mean_amount is None:
                    continue
                
                try:
                    amount = float(mean_amount.text)
                    # Negate the amount for inputs
                    if is_input:
                        amount = -amount
                except (ValueError, TypeError):
                    continue
                
                # Get compartment information if available
                compartment = flow_ref.find('.//common:compartment', ns)
                compartment_text = compartment.text.strip() if compartment is not None else ''
                
                # Get UUID
                uuid = flow_ref.get('refObjectId', '')
                
                exchanges.append({
                    'Flow name': flow_name,
                    'Amount': amount,
                    'Direction': 'Input' if is_input else 'Output',
                    'Compartment': compartment_text,
                    'UUID': uuid,
                    'Process name': base_name
                })
                
            except Exception as e:
                print(f"Error processing exchange in {xml_file}: {str(e)}")
                continue
                
        return exchanges
    
    except Exception as e:
        print(f"Error processing file {xml_file}: {str(e)}")
        return []

def process_xml_directory(directory):
    """
    Process all XML files in a directory and combine results into a single DataFrame.
    Returns a DataFrame containing all exchanges with their process names.
    """
    all_exchanges = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            file_path = os.path.join(directory, filename)
            exchanges = extract_exchanges(file_path)
            all_exchanges.extend(exchanges)
    
    if all_exchanges:
        df = pd.DataFrame(all_exchanges)
        # Save to Excel for reference
        df.to_excel('formatted_lci.xlsx', index=False)
        print(f"Processed data saved to formatted_lci.xlsx ({len(df)} rows)")
        return df
    else:
        print("No exchanges found in the XML files")
        return pd.DataFrame(columns=['Flow name', 'Amount', 'Direction', 'Compartment', 'UUID', 'Process name'])

if __name__ == '__main__':
    # Process XML files in the current directory
    process_xml_directory('.')
