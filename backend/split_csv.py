import os

def split_csv(source_filepath, lines_per_file=130000):
    # Extract the base filename and extension
    filename, ext = os.path.splitext(source_filepath)
    
    # Open the large source file
    with open(source_filepath, 'r', encoding='utf-8') as file:
        # Read and store the header row
        header = file.readline()
        if not header:
            print("The file appears to be empty.")
            return

        file_count = 1
        line_count = 0
        
        # Create and open the first output chunk
        out_file = open(f'{filename}_part{file_count}{ext}', 'w', encoding='utf-8')
        out_file.write(header)

        # Loop through the rest of the file line by line
        for line in file:
            out_file.write(line)
            line_count += 1
            
            # If we hit the 200,000 line limit, cap the file and start a new one
            if line_count == lines_per_file:
                out_file.close()
                print(f"Successfully created: {filename}_part{file_count}{ext}")
                
                file_count += 1
                line_count = 0
                
                # Open the next file and write the header at the top
                out_file = open(f'{filename}_part{file_count}{ext}', 'w', encoding='utf-8')
                out_file.write(header)

        # Close the final file when the loop finishes
        out_file.close()
        print(f"Successfully created: {filename}_part{file_count}{ext}")
        print("Done! Your CSV has been split.")

# ==========================================
# RUNNING THE SCRIPT
# ==========================================
# Replace 'large_data.csv' with the actual name or path of your file.
if __name__ == "__main__":
    split_csv('fraudTest.csv', 130000)
    split_csv('fraudTrain.csv', 130000)