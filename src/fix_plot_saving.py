import os
import json

# Read the notebook
notebook_path = r'e:\3rd_year\6th-sem\await\research\experiments\exp3_dataset_shift_analysis.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with the visualization code
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        # Check if this is the visualization cell
        source = ''.join(cell['source'])
        if 'plt.show()' in source and 'top_features' in source and 'feature in top_features' in source:
            # Found the cell! Now modify it
            new_source = []
            in_loop = False
            indent = '    '
            
            for line in cell['source']:
                # Track when we're in the for loop
                if 'for feature in top_features:' in line:
                    in_loop = True
                    # Add code to create output directory before the loop
                    new_source.append('import os\n')
                    new_source.append('os.makedirs("../../results/figures", exist_ok=True)\n')
                    new_source.append('\n')
                
                # Add the line
                new_source.append(line)
                
                # If we find plt.show(), replace it with savefig
                if in_loop and 'plt.show()' in line:
                    # Remove plt.show()
                    new_source[-1] = ''
                    # Add savefig and close
                    new_source.append(f'{indent}plt.savefig(f"../../results/figures/experiment3_feature_shift_{{feature}}.png", dpi=300, bbox_inches="tight")\n')
                    new_source.append(f'{indent}plt.close()\n')
            
            # Update the cell
            cell['source'] = new_source
            print(f"Modified cell {i}")
            break

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")
print("The plots will be saved to research/results/figures/experiment3_feature_shift_*.png")
