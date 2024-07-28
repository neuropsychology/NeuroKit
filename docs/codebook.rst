Codebook
========

Here you can download the complete codebook which details the variables that you can compute using the NeuroKit package.

.. raw:: html

   <div style="text-align: center;">
       <a href="_static/neurokit_codebook.csv" download="neurokit_codebook.csv">
           <button style="background-color: #4CAF50; color: white; padding: 10px 20px; margin: 10px; border: none; cursor: pointer; width: 50%;">Download Codebook</button>
       </a>
   </div>

This codebook contains detailed descriptions of all variables, their descriptions, and additional metadata.


Codebook Table
==============

.. raw:: html

   <style>
    #csvDataTable {
        width: 100%;
        border-collapse: collapse;
        .. background-color: #f8f8f8;
        color: white;
    }
    #csvDataTable th, #csvDataTable td {
        padding: 8px 12px;
        border: 1px solid #ccc;
        text-align: left;
    }
    </style>

    <div id="csv-table">
        <table id="csvDataTable">
        </table>
    </div>

    <script>

    function parseCSVLine(text) {
        const cols = [];
        let col = '';
        let insideQuotes = false;

        for (let i = 0; i < text.length; i++) {
            const char = text[i];

            if (insideQuotes && char === '"' && text[i + 1] == '"') {
                i++;
                col += char;
                continue;
            }

            if (char === '"' && text[i - 1] !== '\\') {
                insideQuotes = !insideQuotes;
                continue;
            }

            if (char === ',' && !insideQuotes) {
                cols.push(col);
                col = '';
            } else {
                col += char;
            }
        }
        cols.push(col);

        return cols.map(field => field.replace(/""/g, '"')); // Replace escaped quotes
    }

    document.addEventListener("DOMContentLoaded", function() {
        fetch('_static/neurokit_codebook.csv')
            .then(response => response.text())
            .then(csv => {
                let lines = csv.trim().split('\n');
                let html = '<tr><th>' + parseCSVLine(lines[0]).join('</th><th>') + '</th></tr>';
                for (let i = 1; i < lines.length; i++) {
                    html += '<tr><td>' + parseCSVLine(lines[i]).join('</td><td>') + '</td></tr>';
                }
                document.getElementById('csvDataTable').innerHTML = html;
            })
            .catch(error => console.error('Error loading the CSV file:', error));
    });


    </script>

