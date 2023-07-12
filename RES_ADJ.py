# we need to take care of scale, thickness, subtract_right

def res_adjustment(width, height):
    if width >= 1920 and height >= 1080:
        scale = 3
        thickness = 2
        subtract_height = 200
        x = 20
        y = 70
    elif width >= 1280 and height >= 720:
        scale = 2
        thickness = 2
        subtract_height = 120
        x = 20
        y = 50
    else:
        scale = 1
        thickness = 2
        subtract_height = 80
        x = 20
        y = 40
    return (x, y, subtract_height, scale, thickness)

