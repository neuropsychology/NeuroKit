import csv
import os
from docutils import nodes
from docutils.parsers.rst import Directive

abrv_to_sensor = {
            "ecg": "Electrocardiography",
            "eda": "Electrodermal Activity",
            "rsp": "Respiration",
            "ppg": "Photoplethysmography",
            "eeg": "Electroencephalography",
            "emg": "Electromyography",
            "eog": "Electrooculography",
            "hrv": "Heart Rate Variability",
        }

class CSVDocDirective(Directive):
    has_content = True

    def run(self):
        # Codebook path
        csv_file_path = os.path.join(os.path.abspath('.'), "_static", "neurokit_codebook.csv")

        # Check if the file exists and whether it is empty
        file_empty = not os.path.exists(csv_file_path) or os.stat(csv_file_path).st_size == 0

        # List to hold bullet list nodes
        bullet_list = nodes.bullet_list()

        doc_source_name = self.state.document.settings.env.temp_data.get('object')[0]

        maybe_sensor = doc_source_name.split("_")
        doc_sensor = "N/A"

        if len(maybe_sensor) > 0 and maybe_sensor[0] in abrv_to_sensor:
            doc_sensor = abrv_to_sensor[maybe_sensor[0]]

        # Open the CSV file and append the content
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header if file is newly created or empty
            if file_empty:
                header = ['Field Name', 'Field Description', 'Field Category', 'Source File Name']
                writer.writerow(header)

            # Iterate through rows: add them to the codebook and add them to the page
            for line in self.content:

                fields = line.split('|')

                # Remove multi line long space sequences
                for fid in range(len(fields)):
                    fields[fid] = " ".join(fields[fid].split())

                # Append last fields
                fields.append(doc_sensor)
                fields.append(f"{doc_source_name}.py")

                # Write to CSV
                writer.writerow([field.strip() for field in fields])


                # Prepare the documentation stylization
                if len(fields) >= 2:
                    paragraph = nodes.paragraph()

                    # Create backtick formatting around the field name
                    field1 = nodes.literal('', '', nodes.Text(fields[0].strip()))

                    # Add the remainder of the line
                    colon_space = nodes.Text(': ')
                    field2 = nodes.Text(fields[1].strip())

                    # Add all the parts to the paragraph
                    paragraph += field1
                    paragraph += colon_space
                    paragraph += field2

                    # Add to the bullet point list
                    list_item = nodes.list_item()
                    list_item += paragraph
                    bullet_list += list_item

        return [bullet_list]


def setup(app):
    app.add_directive("codebookadd", CSVDocDirective)
