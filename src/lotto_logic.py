from datetime import datetime, timedelta, timezone
import json
import requests
import os
import cv2

class Lotto():
    def __init__(self, ticket) -> None:
        self.nums = []
        self.latest_date = ""
        self.ticket = ticket

    def get_nums(self):
        pass
    def check_nums(self):
        pass

class MegaMillions(Lotto):
    def __init__(self, ticket) -> None:
        super().__init__(ticket)
        self.lotto = "MegaMillions"
    
    def get_nums(self):
        """
        Get the winning numbers
        https://rapidapi.com/avoratechnology/api/mega-millions
        """
        if os.path.exists("./account.json"):
            account = json.load(open("./account.json", "r"))
            now = datetime.now()
            diff_today = now - self.ticket.date
            if self.ticket.date.strftime("%a").lower() == "fri" and diff_today.days > 0 and  diff_today.days < 4 or self.ticket.date.strftime("%a").lower() == "tue" and diff_today.days > 0 and  diff_today.days < 5:
                url = "https://mega-millions.p.rapidapi.com/latest"

                headers = {
                    "X-RapidAPI-Key": account[self.lotto]["API-key"],
                    "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
                }
                #response = requests.request("GET", url, headers=headers)
                #res = response.json()
                res = {"data":[{
                "DrawingDate":"2021-11-09T00:00:00.000Z",
                "FirstNumber":9,
                "SecondNumber":14,
                "ThirdNumber":16,
                "FourthNumber":26,
                "FifthNumber":49,
                "MegaBall":14,
                "Megaplier":3,
                "JackPot":"$45,000,000",
                "NumberSet":"9 14 16 26 49 14 3x"
            }]}
            else:
                q_date = self.ticket.date.strftime("%Y-%m-%d")
                url = "https://mega-millions.p.rapidapi.com/"+q_date

                querystring = {"DrawingDate":q_date}

                headers = {
                    "X-RapidAPI-Key": account[self.lotto]["API-key"],
                    "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
                }

                # response = requests.request("GET", url, headers=headers, params=querystring)
                # res = response.json()
                res = {"data":[{
                "DrawingDate":"2022-08-26T00:00:00.000Z",
                "FirstNumber":"9",
                "SecondNumber":"14",
                "ThirdNumber":"16",
                "FourthNumber":"26",
                "FifthNumber":"49",
                "MegaBall":"14",
                "Megaplier":"3",
                "JackPot":"$45,000,000",
                "NumberSet":"9 14 16 26 49 14 3x"
            }]}

            data = res["data"][0]
            self.nums = data["NumberSet"].split(" ")[0:6]
            with open('nums.json', 'w') as f:
                json.dump({
                    "date": data["DrawingDate"],
                    "nums": self.nums
                }, f)
        else:
            print("Need an account.json file with api keys")

    def check_nums(self):
        #if tix date is older than today then look up numbers
        now = datetime.now()
        diff_today = now - self.ticket.date
        if diff_today.days > 0:
            if os.path.exists("./nums.json"):
                num_file = json.load(open("./nums.json", "r"))
                diff_drawing = now - datetime.strptime(num_file["date"], '%Y-%m-%dT%H:%M:%S.%fZ')
                # if date from saved data = date from ticket
                if diff_today.days == diff_drawing.days:
                    self.nums = num_file["nums"]
                else:
                    self.get_nums()
            else:
                self.get_nums()
        #if tix date hasn't happened yet, return 
        else:
            return "Ticket numbers have not been drawn yet"
        for i in range(0,len(self.ticket.nums)):
            for j in range(0,5):
                if self.ticket.nums[i][j] in self.nums[0:6]:
                    self.ticket.rois[f'col{j+1}']["win"].insert(0,True)
                else:
                    self.ticket.rois[f'col{j+1}']["win"].insert(0,False)
            if self.ticket.nums[i][-1] == self.nums[-1]:
                self.ticket.rois['MegaBall']["win"].insert(0,True)
            else:
                self.ticket.rois['MegaBall']["win"].insert(0,False)
        self.ticket.save_results()
