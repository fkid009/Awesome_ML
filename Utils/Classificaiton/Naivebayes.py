class NaivebayesClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.Xy = pd.concat([X, y], axis = 1)

        self.y_label = y.columns[0]
        self.y_class = y[self.y_label].unique()

        # classify discrete & continuous
        self.con_vars, self.dis_vars = self._classify_variables(X)

        # prior
        self.prior_dict = y[self.y_label].value_counts(normalize=True).to_dict()

        # calculate conditional probaility of discrete variables
        self.dis_likelihood_dict = self._calc_likelihood_dis_var()

        # calculate conditional probaility of conotinious variables
        self.con_likelihood_dict = self._calc_likelihood_con_var()


    def _calc_likelihood_dis_var(self):
        dis_likelihood_dict = dict()
        for label in self.y_class:
            temp_dict1 = dict()
            for dis in self.dis_vars:
                temp_dict2 = dict()
                for var in self.X[dis].unique():
                    temp_dict2[var] = self._conditional_probability(self.Xy, self.y_label, label, dis, var)
                temp_dict1[dis] = temp_dict2
            dis_likelihood_dict[label] = temp_dict1
        return dis_likelihood_dict
    
    def _calc_likelihood_con_var(self):
        con_likelihood_dict = dict()
        for label in self.y_class:
            temp_dict = dict()
            temp_xy = self.Xy[self.Xy[self.y_label] == label]
            for var in self.con_vars:
                mean, std = self._estimate_gaussian(temp_xy, var)
                temp_dict[var] = (mean, std)
            con_likelihood_dict[label] = temp_dict
        return con_likelihood_dict
        

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            temp_dict = dict()
            for j in self.y_class:
                temp_con_dict = self.con_likelihood_dict[j]
                temp_dis_dict = self.dis_likelihood_dict[j]

                temp = 1
                # caculation continuous variable likelihood
                for k in self.con_vars:
                    mean, std = temp_con_dict[k]
                    temp *= self._pdf(mean, std, X.iloc[i][k])
                # calculation discrete variable likelihood
                for k in self.dis_vars:
                    temp *= temp_dis_dict[k][X.iloc[i][k]]

                temp_dict[j] = temp * self.prior_dict[j]
            y_pred.append(self._get_key_of_max_value(temp_dict))

        return np.array(y_pred)

    def _classify_variables(self, df):
        continuous_vars = []
        discrete_vars = []

        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                discrete_vars.append(col)
            else:
                continuous_vars.append(col)
        return continuous_vars, discrete_vars
    
    def _conditional_probability(self, df, event_col, event_val, given_col, given_val):
        """
        데이터 프레임 내에서 조건부 확률 P(event_col=event_val | given_col=given_val) 계산
        
        :param df: pandas DataFrame
        :param event_col: 관심 있는 사건의 컬럼명
        :param event_val: 관심 있는 사건의 값
        :param given_col: 조건으로 주어진 사건의 컬럼명
        :param given_val: 조건으로 주어진 사건의 값
        :return: 조건부 확률 값 (float)
        """
        p_given = len(df[df[given_col] == given_val]) / len(df)

        p_event_given = len(df[(df[event_col] == event_val) & (df[given_col] == given_val)]) / len(df)

        if p_given > 0:
            return p_event_given / p_given
        else:
            return 0  
        
    def _estimate_gaussian(self, df, feature_name):
        mean = df[feature_name].mean()
        std = df[feature_name].std()
        return mean, std
    
    def _pdf(self, mean, var, x):
        numerator = np.exp(-(x - mean) ** 2) / (2 * var)
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _get_key_of_max_value(self, d):
        return max(d, key=d.get) if d else None 