from utils import get_col, load_scaler

class DataManipulation:    
    @staticmethod
    def tidy_data(data):
        #lowercase on column names
        data.columns= data.columns.str.lower()

        #lowercase for values
        for col in data.columns:
            if data[col].dtypes == 'O':
                data[col] = data[col].str.lower()
                
        #resolve missing values
        data = data.fillna(-999)
        
        #drop unused columns
        data.drop([col for col in data.columns if col not in get_col()], axis=1, inplace=True)

        #drop duplicates
        data.drop_duplicates(inplace=True)
        
        #reset index
        data.reset_index(inplace=True, drop=True)

        return data

    @staticmethod
    def encode_gender(data):
        return data.replace({'female': 1, 'male': 2}).astype(int)

    @staticmethod
    def encode_seniorcitizen(data):
        return data.replace({'yes': 1, 'no': 2}).astype(int)

    @staticmethod
    def encode_partner(data):
        return data.replace({'yes': 1, 'no': 2}).astype(int)

    @staticmethod
    def encode_phoneservice(data):
        return data.replace({'yes': 1, 'no': 2}).astype(int)

    @staticmethod
    def encode_multiplelines(data):
        return data.replace({'yes': 1, 'no': 2, 'no phone service': 3}).astype(int)

    @staticmethod
    def encode_contract(data):
        return data.replace({'month-to-month': 1, 'two year': 2, 'one year': 3}).astype(int)

    @staticmethod
    def encode_paymentmethod(data):
        return data.replace({'electronic check': 1, 'mailed check': 2, 'bank transfer (automatic)': 3, 'credit card (automatic)': 4}).astype(int)

    @staticmethod
    def scale_monthlycharges(data):
        scaler = load_scaler('assets/monthlycharges_scaler.save')
        return scaler.transform(data)