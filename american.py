import pandas as pd
import joblib



def prepare_input_data(home_team_code,away_team_code):
    
    input_data = {
        'Home_Team_Code': home_team_code,
        'Away_Team_Code': away_team_code,
    }

    return pd.DataFrame([input_data])  # Return input data as DataFrame

def predict(home_team_name,away_team_name):
    try:
        model = joblib.load('footballmodel_home.pkl')

        team_code= {'Arizona Cardinals': 0, 'Atlanta Falcons': 1, 'Baltimore Colts': 2, 'Baltimore Ravens': '3', 'Boston Patriots': 4, 'Buffalo Bills': 5, 'Carolina Panthers': 6, 'Chicago Bears': 7, 'Cincinnati Bengals': 8, 'Cleveland Browns': 9, 'Dallas Cowboys': 10, 'Denver Broncos': 11, 'Detroit Lions': 12, 'Green Bay Packers': 13, 'Houston Oilers': 14, 'Houston Texans': 15, 'Indianapolis Colts': 16, 'Jacksonville Jaguars': '17', 'Kansas City Chiefs': 18, 'Las Vegas Raiders': 19, 'Los Angeles Chargers': 20, 'Los Angeles Raiders': 21, 'Los Angeles Rams': 22, 'Miami Dolphins': 23, 'Minnesota Vikings': 24, 'New England Patriots': 25, 'New Orleans Saints': 26, 'New York Giants': 27, 'New York Jets': 28, 'Oakland Raiders': 29, 'Philadelphia Eagles': 30, 'Phoenix Cardinals': 31, 'Pittsburgh Steelers': 32, 'San Diego Chargers': 33, 'San Francisco 49ers': 34, 'Seattle Seahawks': 35, 'St. Louis Cardinals': 36, 'St. Louis Rams': 37, 'Tampa Bay Buccaneers': 38, 'Tennessee Oilers': 39, 'Tennessee Titans': 40, 'Washington Commanders': 41, 'Washington Football Team': 42, 'Washington Redskins': 43}
        
        home_team_code = team_code[home_team_name]
        away_team_code= team_code[away_team_name]

        test=prepare_input_data(home_team_code,away_team_code)

        probability= model.predict(test)

        results= {
            home_team_name: probability[0],
            away_team_name: (100 - probability[0])
        }

        return results


    except KeyError as e:
        return {'Error':f"Error: {e} not found. Please check the team code."}
    except FileNotFoundError:
        return {"Error":"Error: File not found."}
    except Exception as e:
        return {'Error':f"An error occurred: {e}"}