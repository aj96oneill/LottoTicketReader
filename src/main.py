from ticket import Ticket
from image_scanner import ImageScanner
from lotto_logic import MegaMillions


if __name__ == "__main__":
    fn = "IMG_2342"
    scanned = ImageScanner().scan(f"./{fn}.png")
    ticket = Ticket(scanned, "MegaMillions")
    # ticket = Ticket("./IMG_2342_output.png", "MegaMillions")
    if not ticket.process_ticket(): print("error")
    MegaMillions(ticket).check_nums()

# TO-DO:
# handle if number on ticket is not found (another yellow box?)
# calculate what was won (multiplier might need to be added to work flow)
# handle other tickets from other lottos
# add draw logic if more than 1 (need an example ticket)

# Done:
# get actual date from ticket
# improve model so number accuracy is up (or improve image quality)
# use date to determine what numbers to pull
# if predicited numbers have low confidence then add a yellow box around them for results.png
# normalize x coord from find_rois or find another way to find area needed
# get the draw 
# make flask endpoint for this