import pygame
import sys
import cv2
import numpy as np
import random
import mediapipe as mp
import datetime
import MPAI4
import os
import time

resolution = 'fhd'
question_count = 10  # 한 게임당 문제 수 설정
pygame.init()
impaths = ["resimgs/bpLogo.png", "resimgs/menuBar.png", "resimgs/playGray.png", "resimgs/playGreen.png", "resimgs/blueline.png", "resimgs/grayline.png", "resimgs/playthanks.png"]
question_time = 10  # 문제당 제한 시간
clock = pygame.time.Clock()
trueans = [0 for i in range(question_count)]
isResult = False
end = False

class Scene:
    def __init__(self, screen):
        self.screen = screen
    
    def update(self):
        pass
    
    def draw(self):
        pass

class homeScene(Scene):
    def __init__(self, screen):
        super().__init__(screen)
        self.load_images()

    def load_images(self):
        # Pre-scaled images
        self.bpLogo = pygame.image.load(impaths[0])
        self.bpLogo = pygame.transform.smoothscale(self.bpLogo, pos(615, 120))
        self.bpLogo_rect = self.bpLogo.get_rect(topleft=pos(100, 150))

        self.startbtn_default = pygame.image.load(impaths[2])
        self.startbtn_default = pygame.transform.smoothscale(self.startbtn_default, pos(30, 33))
        self.startbtn_default_rect = self.startbtn_default.get_rect(topleft=pos(100, 355))

        self.startbtn_hover = pygame.image.load(impaths[3])
        self.startbtn_hover = pygame.transform.smoothscale(self.startbtn_hover, pos(30, 33))

        self.setbtn_default = pygame.image.load(impaths[2])
        self.setbtn_default = pygame.transform.smoothscale(self.setbtn_default, pos(30, 33))
        self.setbtn_default_rect = self.setbtn_default.get_rect(topleft=pos(100, 455))

        self.setbtn_hover = pygame.image.load(impaths[3])
        self.setbtn_hover = pygame.transform.smoothscale(self.setbtn_hover, pos(30, 33))

    def draw(self):
        self.screen.fill((255, 255, 255))

        # Draw the logo and default start button
        self.screen.blit(self.bpLogo, self.bpLogo_rect.topleft)
        
        # Text
        font = pygame.font.Font('Pretendard-Medium.ttf', round(32 * smn))
        sttext = font.render('시작하기', True, (0, 0, 0))
        sttext_rect = sttext.get_rect(topleft=pos(150, 350))
        start_rect = sttext_rect.union(self.startbtn_default_rect)
        self.screen.blit(sttext, pos(150, 350))

        setext = font.render('설정', True, (0, 0, 0))
        setext_rect = setext.get_rect(topleft=pos(150, 450))
        setting_rect = setext_rect.union(self.setbtn_default_rect)
        self.screen.blit(setext, pos(150, 450))

        # Draw the start button
        mouse_pos = pygame.mouse.get_pos()
        if self.startbtn_default_rect.collidepoint(mouse_pos) or sttext_rect.collidepoint(mouse_pos):
            self.screen.blit(self.startbtn_hover, self.startbtn_default_rect.topleft)
        else:
            self.screen.blit(self.startbtn_default, self.startbtn_default_rect.topleft)

        # Draw the setting button
        if self.setbtn_default_rect.collidepoint(mouse_pos) or setext_rect.collidepoint(mouse_pos):
            self.screen.blit(self.setbtn_hover, self.setbtn_default_rect.topleft)
        else:
            self.screen.blit(self.setbtn_default, self.setbtn_default_rect.topleft)

        return {
            'start_clicked': start_rect.collidepoint(mouse_pos),
            'settings_clicked': setting_rect.collidepoint(mouse_pos)
        }

class PoseScene(Scene):
    def __init__(self, screen):
        super().__init__(screen)
        self.load_images()
        self.pose_mode = False
        self.questions, self.answers, self.correct_answers = Qmaster()  # Questions, Answer options, Correct Answers list
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.timer = Timer()

        self.current_Qnum = 0  # Current question number
        self.answer_selected = None  # Track the selected answer
        self.show_answer_result = False  # Flag to display result after answering

    def load_images(self):
        # Pre-scaled images
        self.leftbox_default = pygame.image.load(impaths[5])
        self.leftbox_default = pygame.transform.smoothscale(self.leftbox_default, pos(600, 600))
        self.leftbox_default_rect = self.leftbox_default.get_rect(topleft=pos(25, 87))

        self.leftbox_hover = pygame.image.load(impaths[4])
        self.leftbox_hover = pygame.transform.smoothscale(self.leftbox_hover, pos(600, 600))

        self.rightbox_default = pygame.image.load(impaths[5])
        self.rightbox_default = pygame.transform.smoothscale(self.rightbox_default, pos(600, 600))
        self.rightbox_default_rect = self.rightbox_default.get_rect(topleft=pos(655, 87))

        self.rightbox_hover = pygame.image.load(impaths[4])
        self.rightbox_hover = pygame.transform.smoothscale(self.rightbox_hover, pos(600, 600))

    def draw(self):
        global trueans, isResult
        if self.current_Qnum > question_count-1:
            isResult = True
            return 
        
        if self.correct_answers[self.current_Qnum] >= 2:
            self.pose_mode = True
        else: self.pose_mode = False

        self.screen.fill((243, 243, 243))



        # Draw the camera feed
        camera_width = screen_width
        camera_height = screen_height - 60 * smn
        camera_x = 0
        camera_y = 60 * smn
        camera_rect = pygame.Rect(camera_x, camera_y, camera_width, camera_height)

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return
        
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scaled_frame = scale_to_fit(frame_rgb, camera_width, camera_height)
        cropped_frame = crop_center(scaled_frame, camera_width, camera_height)

        # Pygame Surface로 변환
        frame_surface = pygame.surfarray.make_surface(cropped_frame.swapaxes(0, 1))
        self.screen.blit(frame_surface, camera_rect.topleft)

        # MediaPipe 랜드마크 추출 및 업데이트
        result = self.pose.process(frame_rgb)

        if result.pose_landmarks:
            # 랜드마크 리스트 업데이트
            self.landmarks_list = [(landmark.x, landmark.y, landmark.z) for landmark in result.pose_landmarks.landmark]

        # Menu bar
        try:
            menuBar = pygame.image.load(impaths[1])
            menuBar = pygame.transform.smoothscale(menuBar, pos(1280, 60))
            self.screen.blit(menuBar, pos(0, 0))
        except pygame.error as e:
            print(f"Cannot load image: {impaths[1]}")
            raise SystemExit(e)

        left_selected = False

        if self.show_answer_result:
            # Show result of the answer

            font = pygame.font.Font('Pretendard-Medium.ttf', round(24*smn))
            Qtext = font.render(self.questions[self.current_Qnum], True, (0, 0, 0))
            Qtext_rect = Qtext.get_rect(midtop=pos(640, 15))
            self.screen.blit(Qtext, Qtext_rect)
            Qnumt = font.render(f"{self.current_Qnum + 1}/{question_count}", True, (0, 0, 0))
            Qnumt_rect = Qnumt.get_rect(midtop=pos(70, 15))
            self.screen.blit(Qnumt, Qnumt_rect)

            font = pygame.font.Font('Pretendard-Medium.ttf', round(128*smn))

            if self.answer_selected == self.correct_answers[self.current_Qnum]:
                result_text = font.render("정답!", True, (0, 255, 0))
                trueans[self.current_Qnum] = 1
            else:
                result_text = font.render("오답!", True, (255, 0, 0))

            result_rect = result_text.get_rect(center=(screen_width / 2, screen_height / 2))
            self.screen.blit(result_text, result_rect)

            if self.timer.time().seconds >= 3:
                self.show_answer_result = False
                self.current_Qnum += 1
                self.answer_selected = None
                self.timer.start()

        
        elif self.pose_mode == True:
            print(f'capimgs/pose{self.correct_answers[self.current_Qnum]-1}.png')
            self.crframe = pygame.image.load(f'capimgs/pose{self.correct_answers[self.current_Qnum]-1}.png')
            
            self.crframe = pygame.transform.smoothscale(self.crframe, pos(1280, 720))
            self.crframe_rect = self.crframe.get_rect(topleft=(0, 0))
            self.screen.blit(self.crframe, self.crframe_rect)
            

            print('posemode')

            font = custom_font
            Afont = pygame.font.Font("Pretendard-Medium.ttf",round(80*smn))

            Qtext = font.render(self.questions[self.current_Qnum], True, (0, 0, 0))
            Qtext_rect = Qtext.get_rect(midtop=pos(640, 15))
            self.screen.blit(Qtext, Qtext_rect)

            # Draw the timer text
            calculated_time = question_time - self.timer.time().seconds
            minutes, seconds = divmod(calculated_time, 60)
            formatted_time = f"{minutes:02}:{seconds:02}"

            if question_time - self.timer.time().seconds > 3:
                Ttext = font.render(formatted_time, True, (0, 0, 0))
                Ttext_rect = Ttext.get_rect(midtop=pos(1205, 15))
                self.screen.blit(Ttext, Ttext_rect)
            else:
                Ttext = font.render(formatted_time, True, (255, 0, 0))
                Ttext_rect = Ttext.get_rect(midtop=pos(1205, 15))
                self.screen.blit(Ttext, Ttext_rect)

            # Draw the question number
            Qnumt = font.render(f"{self.current_Qnum + 1}/{question_count}", True, (0, 0, 0))
            Qnumt_rect = Qnumt.get_rect(midtop=pos(70, 15))
            self.screen.blit(Qnumt, Qnumt_rect)


            # Check for time expiration
            if self.timer.time().seconds >= question_time:
                cv2.imwrite('capimgs/capture.png', frame)

                if MPAI4.pose_mode('capimgs/capture.png', f'capimgs/pc{self.correct_answers[self.current_Qnum]-1}.png') == True:
                    self.answer_selected = 2
                else: 
                    self.answer_selected = 0

                self.show_answer_result = True
                self.timer.start()  # Restart timer for result display



        else:
            # Update game elements
            xavg = sum(landmark[0] for landmark in self.landmarks_list)/len(self.landmarks_list)
            left_selected = xavg < 0.5

            # Draw answer boxes
            if left_selected:
                self.screen.blit(self.leftbox_hover, self.leftbox_default_rect.topleft)
            else:
                self.screen.blit(self.leftbox_default, self.leftbox_default_rect.topleft)

            if not left_selected:
                self.screen.blit(self.rightbox_hover, self.rightbox_default_rect.topleft)
            else:
                self.screen.blit(self.rightbox_default, self.rightbox_default_rect.topleft)

            # Draw question text
            font = custom_font
            Afont = pygame.font.Font("Pretendard-Medium.ttf",round(80*smn))

            Qtext = font.render(self.questions[self.current_Qnum], True, (0, 0, 0))
            Qtext_rect = Qtext.get_rect(midtop=pos(640, 15))
            self.screen.blit(Qtext, Qtext_rect)

            LAtext = Afont.render(self.answers[self.current_Qnum][0], True, (0, 0, 0))
            LAtext_rect = LAtext.get_rect(midtop=pos(320, 330))
            self.screen.blit(LAtext, LAtext_rect)

            RAtext = Afont.render(self.answers[self.current_Qnum][1], True, (0, 0, 0))
            RAtext_rect = RAtext.get_rect(midtop=pos(950, 330))
            self.screen.blit(RAtext, RAtext_rect)

            # Draw the timer text
            calculated_time = question_time - self.timer.time().seconds
            minutes, seconds = divmod(calculated_time, 60)
            formatted_time = f"{minutes:02}:{seconds:02}"

            if question_time - self.timer.time().seconds > 3:
                Ttext = font.render(formatted_time, True, (0, 0, 0))
                Ttext_rect = Ttext.get_rect(midtop=pos(1205, 15))
                self.screen.blit(Ttext, Ttext_rect)
            else:
                Ttext = font.render(formatted_time, True, (255, 0, 0))
                Ttext_rect = Ttext.get_rect(midtop=pos(1205, 15))
                self.screen.blit(Ttext, Ttext_rect)

            # Draw the question number
            Qnumt = font.render(f"{self.current_Qnum + 1}/{question_count}", True, (0, 0, 0))
            Qnumt_rect = Qnumt.get_rect(midtop=pos(70, 15))
            self.screen.blit(Qnumt, Qnumt_rect)

            # Check for time expiration
            if self.timer.time().seconds >= question_time:
                self.answer_selected = 0 if left_selected else 1
                self.show_answer_result = True
                self.timer.start()  # Restart timer for result display
class SettingScene(Scene):
    
    def draw(self):
        self.screen.fill((255, 255, 255))
        # Placeholder for settings content
        font = custom_font
        text = font.render('설정 화면', True, (0, 0, 0))
        self.screen.blit(text, pos(150, 250))

class ResultScene(Scene):
        
    def draw(self):
        self.screen.fill((255, 255, 255))
        # Placeholder for settings content
        Pt = pygame.image.load(impaths[6])
        Pt = pygame.transform.smoothscale(Pt, pos(1280, 720))
        self.screen.blit(Pt, pos(0, 0))
        font = pygame.font.Font('Pretendard-Medium.ttf', round(96*smn))
        maintext = font.render(f'{sum(trueans)}/{question_count}', True, (0, 0, 0))
        self.screen.blit(maintext, pos(260, 350))
        maintext2 = font.render(f'{1700+sum(trueans*100)}P', True, (0, 0, 0))
        self.screen.blit(maintext2, pos(800, 350))


class Timer:
    def __init__(self):
        self.start_time = datetime.datetime.now()

    def start(self):
        self.start_time = datetime.datetime.now()

    def time(self):
        return datetime.datetime.now() - self.start_time

def Qmaster():
    question_list = ["[상식 영역] 컴퓨터과학부의 이름은?", "[수학 영역] 1+1은?", "[수학 영역] x->1일 때, (2x-2)(x-3)/(x-1)의 극한값은?", "[국어 영역]'너나안본지두달다돼감'의 올바른 띄어쓰기 방법은 '너 나 안 본 지 두 달 다 돼 감'이다.", "[국어 영역] '그게 되겠니?' vs '그게 돼겠니?'",
                     "뽀롱뽀롱 뽀로로는 남북한 합작 애니메이션이다.",
                     "NewJeans의 맴버 수는 5명이다.",
                     "최근 유행하는 어린이 애니메이션의 제목으로 적절한 것은?",
                     "영화 '기생충'의 개봉 시기는?",
                     "대한민국에서 가장 많은 관객이 본 작품은?",
                     "2024 파리 올림픽에서 대한민국 선수들의 금메달 개수는?",
                     "수능은 매년 11월 셋째 토요일 직전 무슨 요일에 시행?",
                     "르세라핌의 곡 중 가장 최근에 발매된 곡은?",
                     "프로미스나인의 곡으로 알맞은 것은?",
                     "리그 오브 레전드 월드 챔피언십 주제곡으로도 등장한 뉴진스의 곡의 제목은?"
                     ]
    answer_list = [["UBUR", "UBER"], ["1", "2"], ["4", "-4"], ["O", "X"], ["그게 되겠니?", "그게 돼겠니?"],
                   ["O","X"],
                   ["O","X"],
                   ["캐치!티니핑","사랑의 티니핑"],
                   ["2019","2020"],
                   ["명량","신과함께 죄와벌"],
                   ["10","13"],
                   ["수","목"],
                   ["Sweet","Crazy"],
                   ["Supersonic","Supernatural"],
                   ["RISE","GODS"]
                   ]
    true_answers = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]
    #true_answers = [2, 2, 2, 2, 2]

    combined_list = list(zip(question_list, answer_list, true_answers))
    random.shuffle(combined_list)
    selected_list = combined_list[:question_count]
    rqlist, A_list, T_list = zip(*selected_list)

    return list(rqlist), list(A_list), list(T_list)




def pos(x, y):
    """
    relative-positioning function
    """
    return int(x * smn), int(y * smn)

def crop_center(image, crop_width, crop_height):
    """
    이미지 중앙을 기준으로 크롭하는 함수
    :param image: 원본 이미지 (numpy array)
    :param crop_width: 크롭할 너비
    :param crop_height: 크롭할 높이
    :return: 크롭된 이미지
    """
    (h, w) = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    start_x = int(center_x - (crop_width // 2))
    start_y = int(center_y - (crop_height // 2))
    end_x = int(start_x + crop_width)
    end_y = int(start_y + crop_height)
    return image[start_y:end_y, start_x:end_x]

def scale_to_fit(image, target_width, target_height):
    """
    edits image scale
    :param image: 원본 이미지 (numpy array)
    :param target_width: 목표 너비
    :param target_height: 목표 높이
    :return: 스케일 조정된 이미지
    """
    original_height, original_width = image.shape[:2]

    # frame ratio
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # ratio definition
    scale_ratio = max(width_ratio, height_ratio)

    # creating new frame
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    # scale modification
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

# Main game loop
def main():
    global isResult
    global smn, screen_height, screen_width, cap, current_scene, scenes

    # resolution setting
    if resolution == 'hd':
        smn = 1
    elif resolution == 'fhd':
        smn = 1.5
    elif resolution == '4k':
        smn = 3
    screen_height = int(720 * smn)
    screen_width = int(screen_height * 16 / 9)

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("BRAIN POWER")

    # font setting
    font_path = "Pretendard-Medium.ttf" 
    font_size = round(24 * smn)

    # font loading
    try:
        global custom_font
        custom_font = pygame.font.Font(font_path, font_size)
    except FileNotFoundError:
        print(f"Font file not found: {font_path}")
        pygame.quit()
        sys.exit()

    # refresh camera
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        pygame.quit()
        return

    scenes = {
        'home': homeScene(screen),
        'pose': PoseScene(screen),
        'settings': SettingScene(screen),
        'result' : ResultScene(screen)
    }
    
    global current_scene
    current_scene = scenes['home']
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        screen.fill((255, 255, 255))

        result = current_scene.draw()
        current_scene.update()


        if isinstance(current_scene, homeScene):
            if result['start_clicked']:
                current_scene = scenes['pose']
                current_scene.timer.start()
            elif result['settings_clicked']:
                pass  # 설정 화면으로 이동하는 로직 추가 가능

        if isResult:
            current_scene = scenes['result']

        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    main()
