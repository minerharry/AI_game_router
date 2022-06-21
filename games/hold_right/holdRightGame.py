from core.baseGame import RunGame
from PIL import Image, ImageDraw, ImageFont
import math

class HoldRightGame(RunGame):
    def __init__(self,runnerConfig,kwargs):
        super().__init__(runnerConfig,kwargs);
        self.coeff = runnerConfig.rightCoeff;
        self.position = 0.0;

    def processInput(self,inputs):
        self.steps += 1;
        self.position += inputs[0] * self.coeff;

    def renderInput(self,inputs):
        self.processInput(inputs);
        fitness = self.runConfig.fitnessFromGameData(self.getMappedData());
        baseImage = self.getFieldWithAvatar(self.position,(Image.open('images\\mario_avatar.png') if self.steps%2==0 else Image.open('images\\mario_avatar_2.png')),self.runConfig.animation_pixel_per_unit,self.runConfig.animation_unit_per_sign);
        draw = ImageDraw.Draw(baseImage);
        font = ImageFont.truetype('arial.ttf',22);
        draw.text((5,5),'fitness: {0}'.format(fitness),fill=(0,255,0))
        return baseImage;
        
    def getOutputData(self):
        return {"position":self.position};

    def getSignImage(self,text,color=(0,0,0)):
        text = str(text);
        font = ImageFont.truetype('arial.ttf',22);
        signBase = Image.open('images\\sign.jpg');
        signSpace = (58,28);
        yPosition = 3;
        if (len(text)>6 or font.getsize(text)[1]>signSpace[1]):
            font = ImageFont.truetype('arial.ttf',10);
            yPosition = 5;
        width = font.getsize(text)[0];
        xPosition = int((signSpace[0] - width)/2)+1;
        signDrawer = ImageDraw.Draw(signBase);
        signDrawer.text((xPosition,yPosition),text,font=font,fill=color);
        return signBase;
            

    def getSignedField(self,position,pixels_per_unit,units_per_sign):
        background = Image.open('images\\background.jpg');
        decimal_end = '.' in str(units_per_sign);
        field_unit_width = background.width / pixels_per_unit;
        left_unit = position - field_unit_width/2;
        sign_units = [(i+math.ceil(left_unit/units_per_sign))*units_per_sign for i in range(math.ceil(field_unit_width/units_per_sign))];
        sign_positions = [int((unit-left_unit)*pixels_per_unit-30) for unit in sign_units];
        sign_unit_strings = [str(unit) + ('.0' if decimal_end and not '.' in str(unit) else '') for unit in sign_units];
        [background.paste(self.getSignImage(sign_unit_strings[i]),(sign_positions[i],145)) for i in range(len(sign_units))];
        return background;

    def getFieldWithAvatar(self,position, avatar_image, pixels_per_unit, units_per_sign):
        signed_field = self.getSignedField(position,pixels_per_unit,units_per_sign);
        signed_field.paste(avatar_image,(int(signed_field.width/2-avatar_image.width/2),165),avatar_image if (avatar_image.mode == 'RGBA') else None);
        return signed_field;