import pandas as pd
from cmath import sqrt
from collections import Counter

class Problem1Data:
    def __init__(self) -> None:
        pass

class Problem2Data:
    def __init__(self) -> None:
        self.h_a1: float
        self.h_a2: float
        self.d_geom_a1: float
        self.d_geom_a2: float

class Problem3Data:
    def __init__(self) -> None:
        pass

class HomeworkData:
    def __init__(self) -> None:
        self.problem1_data: Problem1Data
        self.problem2_data: Problem2Data
        self.problem3_data: Problem3Data

class Problem1:
    def __init__(self) -> None:
        # excel_file_path = 'Lab2_Trinc.Emanuel.xls'
        self.excel_file_path = "Lab1_First.Last.xls"
        self.sheet_name = 'RND1'
        self.total_points = 3
        self.row_index = 5
    
    def assess_problem(self, stud_number) -> float:
        # Generate antenna height array [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        # and at the same time add the stud_number to each array element (2 rows in the excel table).
        data = self._extract_problem_data()
        height_list = [height + stud_number for height in range(10, 24)]
        d_geom = [(3.57 * sqrt(height)) for height in height_list]
        matches = []
        for i in range(0, len(height_list)):
            if (d_geom[i] == data[i]):
                matches.append(True)
            else:
                matches.append(False)
        right_answers = Counter(matches)[True]
        return 3/14 * right_answers

    def _extract_problem_data(self) -> []:
        """
        """
        # Read an Excel file into a pandas DataFrame
        df: pd.DataFrame = None
        try:
            df = pd.read_excel(self.excel_file_path, sheet_name=self.sheet_name)
            # Check DataFrame Shape
            print("Dataframe shape: " + str(df.shape))
            print(df.head())
            if df.empty:
                print("DataFrame is empty (1st row on excel is regarded as the head).")
            else:
                print("DataFrame loaded successfully.")
                # Further processing or analysis can be performed here
        except Exception as e:
            print(f"Error loading DataFrame: {str(e)}")
        
        data = []
        for col_index in range(1, 15):
            cell_value = df.iloc[self.row_index, col_index]
            data.append(cell_value)
        return data

class Problem2:
    def __init__(self) -> None:
        self.excel_file_path = "test.xls"
        self.sheet_name = 'RND2'
        self.total_points = 3
        self.col_index_h = 1
        self.col_index_d_geom = 2
        self.row_index_a1 = 9
        self.row_index_a2 = 10
    
    def assess_problem(self, stud_number) -> float:
        """
        """
        data: Problem2Data = self._extract_problem_data()

        h_a1 = 20 + stud_number
        h_a2 = 20 + stud_number
        #assert h_a1 == data.h_a1
        #assert h_a2 == data.h_a2

        d_geom_a1 = 3.57 * sqrt(data.h_a1)
        d_geom_a2 = 3.57 * sqrt(data.h_a2)

        grade = 0
        if d_geom_a1 == data.d_geom_a1:
            grade = grade + 1.5
        if d_geom_a2 == data.d_geom_a2:
            grade = grade + 1.5
        return grade

    def _extract_problem_data(self) -> Problem2Data:
        """
        """
        # Read an Excel file into a pandas DataFrame
        df: pd.DataFrame = None
        try:
            df = pd.read_excel(self.excel_file_path, sheet_name=self.sheet_name)
            # Check DataFrame Shape
            print("Dataframe shape: " + str(df.shape))
            print(df.head())
            if df.empty:
                print("DataFrame is empty (1st row on excel is regarded as the head).")
            else:
                print("DataFrame loaded successfully.")
                # Further processing or analysis can be performed here
        except Exception as e:
            print(f"Error loading DataFrame: {str(e)}")
        
        data = Problem2Data()
        data.h_a1 = df.iloc[self.row_index_a1, self.col_index_h]
        data.h_a2 = df.iloc[self.row_index_a2, self.col_index_h]
        data.d_geom_a1 = df.iloc[self.row_index_a1, self.col_index_d_geom]
        data.d_geom_a2 = df.iloc[self.row_index_a2, self.col_index_d_geom]
        
        return data

problem1 = Problem1()
gradePb1 = problem1.assess_problem(1)
print("Grade P1: " + str(gradePb1))

problem2 = Problem2()
gradePb2 = problem2.assess_problem(1)
print("Grade P2: " + str(gradePb2))