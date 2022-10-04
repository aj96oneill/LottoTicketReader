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
    
    def get_nums(self):
        """
        Get the winning numbers
        """
        if os.path.exists("./account.json"):
            account = json.load(open("./account.json", "r"))
            url = "https://mega-millions.p.rapidapi.com/latest"

            headers = {
                "X-RapidAPI-Key": account["MegaMillions"]["API-key"],
                "X-RapidAPI-Host": "mega-millions.p.rapidapi.com"
            }
            # print(datetime.strptime("2022-09-30T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ'))
            # print(datetime.now(timezone.utc))
            # print(datetime.now() - datetime.strptime("2022-09-30T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ'))

            response = {"data":[{}]}#requests.request("GET", url, headers=headers)

            # status:"success"
            # data:[{
            #     DrawingDate:"2021-11-09T00:00:00.000Z"
            #     FirstNumber:9
            #     SecondNumber:14
            #     ThirdNumber:16
            #     FourthNumber:26
            #     FifthNumber:49
            #     MegaBall:14
            #     Megaplier:3
            #     JackPot:"$45,000,000"
            #     NumberSet:"9 14 16 26 49 14 3x"
            # }]
            #print(response)
            data = response["data"][0]
            # self.nums = [data["FirstNumber"], data["SecondNumber"], data["ThirdNumber"], data["FourthNumber"], data["FifthNumber"], data["MegaBall"]]
            self.nums = ["9", "14", "16", "26", "49", "14"]
        else:
            print("Need an account.json file with api keys")

    def check_nums(self):
        if len(self.nums) == 0: self.get_nums()
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

if __name__ == "__main__":
    ticket = ""
    lotto = MegaMillions(ticket)
    lotto.get_nums()
    print(lotto.nums)