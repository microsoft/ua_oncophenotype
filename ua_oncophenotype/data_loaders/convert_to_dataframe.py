import os
import xml.etree.ElementTree as ET

import pandas as pd


class LoadTextDataFrame:
    def __init__(self, path: str, text_column: str) -> None:
        """
        Initializes the dataframe loader.
        @path: path to the file or directory
        @text_column: name of the text column
        """
        self.text_column = text_column

        _, file_extension = os.path.splitext(path)
        if file_extension.lower() == ".json":
            self.df = self.load_json(path)
        elif file_extension.lower() == ".csv":
            self.df = self.load_csv(path)
        elif file_extension.lower() == ".xml":
            self.df = self.load_xml(path)
        elif os.path.isdir(path):
            # check if directory contains json or csv files
            files = os.listdir(path)
            filtered_files = [
                os.path.join(path, file)
                for file in files
                if os.path.splitext(file)[1].lower() in [".json", ".csv", ".xml"]
            ]
            if len(filtered_files) == 0:
                Warning(
                    "Directory does not contain any json, csv or xml files. DataFrame"
                    " will be empty"
                )
            self.df = self.load_dir(filtered_files)
        else:
            raise ValueError("Invalid file type.")

    def load_json(self, path: str) -> pd.DataFrame:
        """
        Loads a json file from the given path.
        @path: path to the json file
        returns: pandas dataframe
        """
        df = pd.read_json(path)
        return df

    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Loads a csv file from the given path.
        @path: path to the csv file
        returns: pandas dataframe
        """
        df = pd.read_csv(path)
        return df

    def load_xml(self, path: str) -> pd.DataFrame:
        """
        Loads an xml file from the given path.
        @path: path to the xml file
        returns: pandas dataframe
        """

        def parse_node(node, d):
            for element in node:
                if len(element) == 0:
                    if element.text:
                        d[element.tag] = element.text
                    for attrib in element.attrib:
                        d[element.tag + "_" + attrib] = element.attrib[attrib]
                else:
                    d = parse_node(element, d)
            return d

        tree = ET.parse(path)
        root = tree.getroot()

        data_dict = parse_node(root, {})

        data = [data_dict]
        df = pd.DataFrame(data)
        return df

    def load_dir(self, file_names: list) -> pd.DataFrame:
        """
        Loads all csv, json or xml files from the given directory.
        @file_names: files in the directory
        returns: pandas dataframe
        """
        dfs = []
        for f in file_names:
            ext = os.path.splitext(f)[1].lower()
            if ext == ".json":
                dfs.append(self.load_json(f))
            elif ext == ".csv":
                dfs.append(self.load_csv(f))
            elif ext == ".xml":
                dfs.append(self.load_xml(f))
        df = pd.concat(dfs)

        return df
