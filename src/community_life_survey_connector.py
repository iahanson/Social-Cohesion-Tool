"""
Community Life Survey Data Connector
Handles ingestion of Community Life Survey data with complex sheet structure
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class CommunityLifeSurveyConnector:
    """Connector for Community Life Survey data"""
    
    def __init__(self, file_path: str = "data/Community_Life_Survey_2023_24.xlsx", use_cleaned: bool = True):
        # Use cleaned file by default if available
        cleaned_file = file_path.replace('.xlsx', '_cleaned.xlsx')
        
        if use_cleaned and os.path.exists(cleaned_file):
            self.file_path = cleaned_file
            self.use_cleaned_format = True
            print(f"✅ Using cleaned Community Life Survey data: {cleaned_file}")
        else:
            if use_cleaned:
                print(f"⚠️ Cleaned file not found: {cleaned_file}, using original file")
            self.file_path = file_path
            self.use_cleaned_format = False
        self.survey_data = {}
        self.processed_data = None
        self.lad_mapping = {}
        self._data_loaded = False  # Cache flag to avoid reloading
        
    def load_all_sheets(self) -> Dict[str, pd.DataFrame]:
        """Load all sheets from the Community Life Survey file"""
        # Return cached data if already loaded
        if self._data_loaded and self.survey_data:
            print("📊 Using cached Community Life Survey data")
            return self.survey_data
            
        try:
            if not os.path.exists(self.file_path):
                print(f"❌ Community Life Survey file not found: {self.file_path}")
                return {}
            
            print(f"📊 Loading Community Life Survey data from: {self.file_path}")
            xl = pd.ExcelFile(self.file_path)
            
            print(f"Found {len(xl.sheet_names)} sheets")
            
            loaded_sheets = {}
            for i, sheet_name in enumerate(xl.sheet_names):
                print(f"Loading sheet {i+1}/{len(xl.sheet_names)}: {sheet_name}")
                try:
                    df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None)
                    loaded_sheets[sheet_name] = df
                except Exception as e:
                    print(f"⚠️ Error loading sheet '{sheet_name}': {e}")
            
            self.survey_data = loaded_sheets
            self._data_loaded = True  # Mark as loaded
            print(f"✅ Successfully loaded {len(loaded_sheets)} sheets")
            return loaded_sheets
            
        except Exception as e:
            print(f"❌ Error loading Community Life Survey data: {e}")
            return {}
    
    def analyze_sheet_structure(self, sheet_name: str) -> Dict[str, Any]:
        """Analyze the structure of a specific sheet"""
        if sheet_name not in self.survey_data:
            return {"error": "Sheet not found"}
        
        df = self.survey_data[sheet_name]
        
        analysis = {
            "sheet_name": sheet_name,
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "data_start_row": 9,  # Data starts at row 10 (0-indexed = 9)
            "question": None,
            "lad_column": 1,  # Column B (0-indexed = 1)
            "data_columns": [],
            "sample_lads": []
        }
        
        # Get question from Column A, Line 9 (0-indexed = 8)
        if len(df) > 8:
            question_cell = df.iloc[8, 0]  # Row 9, Column A
            analysis["question"] = str(question_cell) if pd.notna(question_cell) else None
        
        # Get sample LAD names from Column B, starting from Line 10 (0-indexed = 9)
        if len(df) > 9:
            # Get LAD names from Column B (index 1), starting from row 10
            lad_start_row = 9  # Row 10 (0-indexed)
            lad_end_row = min(300, len(df))  # Up to row 300 or end of data
            
            sample_lads = []
            for row_idx in range(lad_start_row, lad_end_row):
                lad_cell = df.iloc[row_idx, 1]  # Column B
                if pd.notna(lad_cell) and str(lad_cell).strip():
                    sample_lads.append(str(lad_cell).strip())
                    if len(sample_lads) >= 10:  # Get first 10 LADs
                        break
            
            analysis["sample_lads"] = sample_lads
        
        # Get data columns from row 10 (0-indexed = 9)
        if len(df) > 9:
            header_row = df.iloc[9].tolist()  # Row 10 headers
            analysis["data_columns"] = [col for col in header_row if pd.notna(col)]
        
        return analysis
    
    def extract_sheet_data(self, sheet_name: str) -> Optional[pd.DataFrame]:
        """Extract structured data from a specific sheet"""
        if sheet_name not in self.survey_data:
            return None
        
        df = self.survey_data[sheet_name]
        
        # Handle cleaned data format
        if self.use_cleaned_format:
            # For cleaned data, first row is headers, data starts from second row
            try:
                # Use first row as headers
                headers = df.iloc[0].tolist()
                
                # Create DataFrame starting from second row
                data_df = df.iloc[1:].copy()
                data_df.columns = headers
                
                # Clean up the data - remove rows where LAD name (Column B) is missing
                if len(headers) > 1:
                    lad_column = headers[1]  # Column B should be LAD names
                    data_df = data_df.dropna(subset=[lad_column])
                
                # Add metadata
                data_df['question'] = sheet_name  # Use sheet name as question for cleaned data
                data_df['sheet_name'] = sheet_name
                
                return data_df
                
            except Exception as e:
                print(f"❌ Error extracting cleaned data from sheet '{sheet_name}': {e}")
                return None
        else:
            # Original format - data starts at row 10 (0-indexed = 9)
            data_start_row = 9
            
            if len(df) <= data_start_row:
                print(f"⚠️ Sheet '{sheet_name}' has insufficient data (only {len(df)} rows)")
                return None
            
            # Extract the data
            try:
                # Get headers from row 10 (0-indexed = 9)
                headers = df.iloc[data_start_row].tolist()
                
                # Create a new DataFrame starting from row 11 (0-indexed = 10)
                data_df = df.iloc[data_start_row+1:].copy()
                data_df.columns = headers
                
                # Clean up the data - remove rows where LAD name (Column B) is missing
                if len(headers) > 1:
                    lad_column = headers[1]  # Column B should be LAD names
                    data_df = data_df.dropna(subset=[lad_column])
                
                # Add metadata
                question = df.iloc[8, 0] if len(df) > 8 else sheet_name  # Row 9, Column A
                data_df['question'] = str(question) if pd.notna(question) else sheet_name
                data_df['sheet_name'] = sheet_name
                
                return data_df
                
            except Exception as e:
                print(f"❌ Error extracting data from sheet '{sheet_name}': {e}")
                return None
    
    def process_all_sheets(self) -> pd.DataFrame:
        """Process all sheets and combine into a single dataset"""
        if not self.survey_data:
            self.load_all_sheets()
        
        if not self.survey_data:
            return pd.DataFrame()
        
        print("🔄 Processing all sheets...")
        
        all_data = []
        processed_count = 0
        
        for sheet_name in self.survey_data.keys():
            print(f"Processing sheet: {sheet_name}")
            
            # Analyze structure first
            analysis = self.analyze_sheet_structure(sheet_name)
            if "error" in analysis:
                continue
            
            # Extract data
            sheet_data = self.extract_sheet_data(sheet_name)
            if sheet_data is not None and not sheet_data.empty:
                all_data.append(sheet_data)
                processed_count += 1
                print(f"  ✅ Extracted {len(sheet_data)} rows")
            else:
                print(f"  ⚠️ No data extracted")
        
        if all_data:
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"✅ Combined data from {processed_count} sheets: {len(combined_data)} total rows")
            
            self.processed_data = combined_data
            return combined_data
        else:
            print("❌ No data could be extracted from any sheets")
            return pd.DataFrame()
    
    def create_lad_mapping(self) -> Dict[str, str]:
        """Create mapping from LAD names to LAD codes"""
        if self.processed_data is None:
            return {}
        
        # Get unique LAD names
        lad_names = self.processed_data.iloc[:, 1].dropna().unique()  # Second column should be LAD names
        
        print(f"📋 Found {len(lad_names)} unique LAD names")
        print("Sample LAD names:")
        for i, name in enumerate(lad_names[:10]):
            print(f"  {i+1}. {name}")
        
        # Create a simple mapping (in a real implementation, you'd use official LAD codes)
        lad_mapping = {}
        for name in lad_names:
            # Simple mapping - in practice you'd use official LAD codes
            lad_mapping[name] = name.replace(" ", "_").upper()
        
        self.lad_mapping = lad_mapping
        return lad_mapping
    
    def get_question_summary(self) -> Dict[str, Any]:
        """Get summary of all questions in the survey"""
        if self.processed_data is None:
            return {}
        
        questions = self.processed_data['question'].value_counts()
        sheets = self.processed_data['sheet_name'].value_counts()
        
        return {
            "total_questions": len(questions),
            "total_sheets": len(sheets),
            "total_responses": len(self.processed_data),
            "questions": questions.to_dict(),
            "sheets": sheets.to_dict()
        }
    
    def get_lad_data(self, lad_name: str) -> pd.DataFrame:
        """Get all data for a specific Local Authority District"""
        if self.processed_data is None:
            return pd.DataFrame()
        
        # Filter data for the specific LAD
        lad_data = self.processed_data[self.processed_data.iloc[:, 1] == lad_name]
        return lad_data
    
    def get_question_data(self, question: str) -> pd.DataFrame:
        """Get all data for a specific question"""
        if self.processed_data is None:
            return pd.DataFrame()
        
        # Filter data for the specific question
        question_data = self.processed_data[self.processed_data['question'] == question]
        return question_data
    
    def export_processed_data(self, output_file: str = "data/community_life_survey_processed.csv"):
        """Export processed data to CSV"""
        if self.processed_data is None:
            print("❌ No processed data to export")
            return
        
        try:
            self.processed_data.to_csv(output_file, index=False)
            print(f"✅ Exported processed data to: {output_file}")
        except Exception as e:
            print(f"❌ Error exporting data: {e}")

def main():
    """Test the Community Life Survey connector"""
    connector = CommunityLifeSurveyConnector()
    
    # Load all sheets
    sheets = connector.load_all_sheets()
    
    if sheets:
        # Analyze first few sheets
        print("\n🔍 Analyzing first 3 sheets:")
        for i, sheet_name in enumerate(list(sheets.keys())[:3]):
            print(f"\nSheet {i+1}: {sheet_name}")
            analysis = connector.analyze_sheet_structure(sheet_name)
            print(f"  Data start row: {analysis.get('data_start_row', 'Not found')}")
            print(f"  Question: {analysis.get('question', 'Not found')}")
            print(f"  Sample LADs: {analysis.get('sample_lads', [])[:3]}")
        
        # Process all sheets
        processed_data = connector.process_all_sheets()
        
        if not processed_data.empty:
            # Get summary
            summary = connector.get_question_summary()
            print(f"\n📊 Summary:")
            print(f"  Total questions: {summary.get('total_questions', 0)}")
            print(f"  Total sheets: {summary.get('total_sheets', 0)}")
            print(f"  Total responses: {summary.get('total_responses', 0)}")
            
            # Export processed data
            connector.export_processed_data()

if __name__ == "__main__":
    main()
