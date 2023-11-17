# Data Cleaning CLI

## Description

This repository contains some example code for creating data cleaning pipelines using the Rich Library in Python [https://rich.readthedocs.io/en/latest/](https://rich.readthedocs.io/en/latest/).

To use the library, you need to run `pip install rich` in your terminal.

Rich let's you create a CLI with a lot of formatting options.

![Example of a CLI created with Rich](https://raw.githubusercontent.com/Textualize/rich-cli/main/imgs/csv1.png)

Some general principles for creating a CLI for data cleaning:

- Try to use the formatting to make the CLI as readable as possible (things like Table and Panel are good for this)
- Include a count of how many items you have left to clean (so you don't get discouraged)
- Write the output to a file as you are cleaning (that way if you have to stop the process, you don't lose your work)

### Script and Data

The script in this repository is very basic but provides an example of how you might create a CLI to clean data. The data in this case is scraped data from iSchool website around faculty and staff.