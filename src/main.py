from ticket import Ticket
from image_scanner import ImageScanner
from lotto_logic import MegaMillions

if __name__ == "__main__":
    fn = "IMG_2342"
    image_scan = ImageScanner()
    scanned = image_scan.scan(f"./{fn}.png")
    ticket = Ticket(scanned, "MegaMillions")
    if not ticket.process_ticket(): print("error")
    MegaMillions(ticket).check_nums()

# TO-DO:
# get actual date from ticket
# use date to determine what numbers to pull
# improve model so number accuracy is up (or improve image quality)

# handle if number on ticket is not found/ super low confidence? (yellow box)

# calculate the what was won (multiplier might need to be added to work flow)

# handle other tickets from other lottos

# make flask endpoint for this