# Open the text file in read mode
with open('filtered-data/filtered-fen.pgn', 'r') as file:
    # Open a new file to write the extracted substrings
    with open('filtered-data/fens.txt', 'w') as extracted_file:
        # Iterate through each line in the file
        for line in file:
            # Check if the line contains '{' and '}'
            open_bracket_index = line.find('{')
            if open_bracket_index != -1:
                # Find the substring between quotes
                start_quote_index = open_bracket_index + 2
                end_quote_index = line.find('"', start_quote_index + 1)

                if end_quote_index != -1:
                    # Write the substring to the new file as a new line
                    extracted_file.write(line[start_quote_index + 1 : end_quote_index] + '\n')
