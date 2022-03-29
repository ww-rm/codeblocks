#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <Windows.h>
#include <time.h>
#define random(n) (rand() % n + 1)
#define DIRECTIONKEY -32
#define UP 72
#define LEFT 75
#define RIGHT 77
#define DOWN 80

HANDLE screen_out = GetStdHandle(STD_OUTPUT_HANDLE); //获取标准输出台
CONSOLE_CURSOR_INFO cursor_info = {25, 0};           //光标信息
COORD position = {0, 0};                             //光标位置

int COLUMN = 30; //区域行数
int ROW = 25;    //区域列数

int RAW_X = 16; //游戏区域偏移量
int RAW_Y = 1;  //

struct SnakeBody //蛇身
{
    int x;           //坐标
    int y;           //
    SnakeBody *last; //上一个
    SnakeBody *next; //下一个
};

struct SnakeHead //蛇头
{
    int x;           //坐标
    int y;           //
    int direction;   //前进方向
    int speed;       //前进速度
    int length;      //长度，包括头
    int score;       //得分
    SnakeBody *next; //蛇身
};

SnakeHead *head; //蛇头指针
SnakeBody *rear; //蛇尾指针

FILE *rank;

int getHighestScore()
{
    int score = 0;
    rank = fopen("greedysnake.dat", "r");
    if (rank != NULL)
    {
        fread(&score, sizeof(int), 1, rank);
    }
    fclose(rank);
    return score;
}

void updateHighestScore()
{
    int history = getHighestScore();
    if (head->score > history)
    {
        rank = fopen("greedysnake.dat", "w");
        fwrite(&head->score, sizeof(int), 1, rank);
        fclose(rank);
    }
}

int FOOD_X; //食物坐标
int FOOD_Y; //
bool FOOD;  //当前食物是否存在

int selectgamemodenumber()
{
	char key;
	int modenumber;
	system("cls");
	SetConsoleCursorInfo(screen_out, &cursor_info);
	position.X=COLUMN-1+RAW_X-8;
	position.Y=ROW/2+RAW_Y-2;
	SetConsoleCursorPosition(screen_out, position);
	printf("↑↓选择，Enter进入游戏"); 
	position.X=COLUMN-1+RAW_X;
	position.Y=ROW/2+RAW_Y;
	SetConsoleCursorPosition(screen_out, position);
	printf("挑战模式");
	position.Y+=1;
	SetConsoleCursorPosition(screen_out, position);
	printf("围墙模式"); 
	position.X-=4;
	position.Y-=1;
	SetConsoleCursorPosition(screen_out, position); 
	printf("→");
	modenumber=1;
	while(true)
	{
		if (_kbhit())
		{
			key=getch();
			if(key==13)
			{
				system("cls");
				return modenumber;
			}
			else if(key == DIRECTIONKEY)
			{
				key=getch();
				if(key==UP)
				{
					SetConsoleCursorPosition(screen_out, position); 
					printf("  ");
					int temp=modenumber;
					if(modenumber==1)position.Y+=1;
					else position.Y-=1;
					SetConsoleCursorPosition(screen_out, position); 
					printf("→");
					if(modenumber==1)
					modenumber=2;
					else modenumber=1; 
				}
				else if(key==DOWN)
				{
					SetConsoleCursorPosition(screen_out, position); 
					printf("  ");
					int temp=modenumber;
					if(modenumber==2)position.Y-=1;
					else position.Y+=1;
					SetConsoleCursorPosition(screen_out, position); 
					printf("→");
					if(modenumber==1)
					modenumber=2;
					else modenumber=1; 
				}
			}
			else continue;
		}
	}
}

void initScreen() //输出一个框
{
    system("cls");                                  //清屏
    char wall[4] = "□";                             //墙用□围起来
    SetConsoleCursorInfo(screen_out, &cursor_info); //设置光标属性，隐藏
    position.X = -1 + RAW_X;                        //初始化坐标加偏移量
    position.Y = 0 + RAW_Y;                         //
    for (int i = 0; i <= COLUMN; i++)               //上
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.X += 2;
    }
    for (int i = 0; i <= ROW; i++) //右
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.Y++;
    }
    for (int i = 0; i <= COLUMN; i++) //下
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.X -= 2;
    }
    for (int i = 0; i <= ROW; i++) //左
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.Y--;
    }
    position.X = 2 * COLUMN + 3 + RAW_X; //输出得分区域
    position.Y = 0 + RAW_Y;
    for (int i = 0; i <= 8; i++)
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.X += 2;
    }
    for (int i = 0; i <= (int)ROW / 2 - 1; i++)
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.Y++;
    }
    for (int i = 0; i <= 8; i++)
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.X -= 2;
    }
    for (int i = 0; i <= (int)ROW / 2 - 1; i++)
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.Y--;
    }
    position.X += 2;
    position.Y += (int)ROW / 8 - 1;
    SetConsoleCursorPosition(screen_out, position);
    printf("    当前得分");
    position.Y += 2;
    SetConsoleCursorPosition(screen_out, position);
    printf("       00");
    position.Y += 2;
    SetConsoleCursorPosition(screen_out, position);
    printf("    历史最高");
    position.Y += 2;
    SetConsoleCursorPosition(screen_out, position);
    printf("       %02d", getHighestScore()); //获取历史最高分
    position.X = 2 * COLUMN + 3 + RAW_X;      //输出提示框
    position.Y = (int)ROW / 2 + 1 + RAW_Y;
    for (int i = 0; i <= 8; i++)
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.X += 2;
    }
    for (int i = 0; i <= (int)ROW / 2; i++)
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.Y++;
    }
    for (int i = 0; i <= 8; i++)
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.X -= 2;
    }
    for (int i = 0; i <= (int)ROW / 2; i++)
    {
        SetConsoleCursorPosition(screen_out, position);
        printf(wall);
        position.Y--;
    }
    position.X += 2;
    position.Y += 2;
    SetConsoleCursorPosition(screen_out, position);
    printf(" 任意键开始游戏");
    position.Y += 2;
    SetConsoleCursorPosition(screen_out, position);
    printf(" 方向键控制移动");
    position.Y += 2;
    SetConsoleCursorPosition(screen_out, position);
    printf("'p'暂停/返回游戏");
    position.Y += 2;
    SetConsoleCursorPosition(screen_out, position);
    printf("  'q'退出游戏");
}

void initSnake(int speed = 200, int length = 4, int raw_x = COLUMN / 2, int raw_y = ROW / 2)	//初始化蛇 
{
    SnakeBody *p;
    head = (SnakeHead *)malloc(sizeof(SnakeHead));
    rear = (SnakeBody *)malloc(sizeof(SnakeBody));
    head->direction = RIGHT; //默认右方向
    head->speed = speed;     //默认速度 1格/200毫秒
    head->length = length;   //初始长度为4
    head->score = 0;         //得分为零
    head->next = rear;       //指向下一个
    head->x = raw_x;         //设置初始蛇头位置
    head->y = raw_y;         //
    rear->last = NULL;       //置空
    rear->next = NULL;       //
    rear->x = raw_x - 1;     //设置初始值
    rear->y = raw_y;         //
    for (int i = 0; i <= length - 3; i++)
    {
        p = rear;
        rear->next = (SnakeBody *)malloc(sizeof(SnakeBody));
        rear = rear->next;
        p->next = rear;
        rear->last = p;
        rear->next = NULL;
        rear->x = p->x - 1;
        rear->y = p->y;
    }
}

void delSnake()			//删除蛇 
{
    SnakeBody *p;
    while (rear != NULL) //删除蛇身
    {
        p = rear;
        rear = rear->last;
        free(p);
    }
    free(head); //删除蛇头
}

void printSnake()
{
    SnakeBody *p = head->next;
    position.X = 2 * head->x - 1 + RAW_X;           //蛇头坐标加偏移量
    position.Y = head->y + RAW_Y;                   //
    SetConsoleCursorPosition(screen_out, position); //设置蛇头坐标
    printf("◎");                                    //蛇头
    while (p != NULL)                               //输出蛇身
    {
        position.X = 2 * p->x - 1 + RAW_X;
        position.Y = p->y + RAW_Y;
        SetConsoleCursorPosition(screen_out, position);
        printf("○");
        p = p->next;
    }
}

void clearSnake(int speed = 0)						//清楚蛇身 
{
    SnakeBody *p = head->next;
    position.X = 2 * head->x - 1 + RAW_X;           //蛇头坐标加偏移量
    position.Y = head->y + RAW_Y;                   //
    SetConsoleCursorPosition(screen_out, position); //设置蛇头坐标
    putchar(' ');                                   //清除蛇头
    putchar(' ');
    while (p != NULL)                               //清除蛇身
    {
        position.X = 2 * p->x - 1 + RAW_X;
        position.Y = p->y + RAW_Y;
        SetConsoleCursorPosition(screen_out, position);
        Sleep(speed);
        putchar(' ');
        putchar(' ');
        p = p->next;
    }
}

void _lengthenSnake() //加长一格蛇身
{
    SnakeBody *p = head->next;
    head->length++;                                      //长度加一
    head->next = (SnakeBody *)malloc(sizeof(SnakeBody)); //将蛇头位置变为蛇身
    head->next->last = NULL;                             //初始化新蛇身
    head->next->x = head->x;                             //
    head->next->y = head->y;                             //
    head->next->next = p;                                //连接原蛇身
    p->last = head->next;                                //
    switch (head->direction)                             //更改蛇头坐标
    {
    case UP:
        head->y = ((head->y - 2 + ROW) % ROW) + 1;
        break;
    case RIGHT:
        head->x = (head->x % COLUMN) + 1;
        break;
    case DOWN:
        head->y = (head->y % ROW) + 1;
        break;
    case LEFT:
        head->x = ((head->x - 2 + COLUMN) % COLUMN) + 1;
        break;
    }
}

void updateScore()	//更新分数 
{
    position.X = 2 * COLUMN + 5 + RAW_X;
    position.Y = 2 + (int)ROW / 8 - 1 + RAW_Y;
    SetConsoleCursorPosition(screen_out, position);
    printf("       %02d", head->score);
}

void _moveSnake()
{
    _lengthenSnake();
    head->length--;
    SnakeBody *p = rear; //删除原蛇尾
    rear = rear->last;   //尾指针前移
    rear->next = NULL;   //置空
    free(p);
}

void changeDirection(char direction) //改变蛇头方向
{
    switch (direction)
    {
    case UP:
        if (head->direction != DOWN)
        {
            head->direction = UP;          
        }
        break;
    case DOWN:
        if (head->direction != UP)
        {
            head->direction = DOWN;
        }
        break;
    case LEFT:
        if (head->direction != RIGHT)
        {
            head->direction = LEFT;
        }
        break;
    case RIGHT:
        if (head->direction != LEFT)
        {
            head->direction = RIGHT;
        }
        break;
    }
}

void createFood() //创建随机食物
{
    srand(time(NULL)); //设置随机种子
    SnakeBody *p;
    SnakeHead *p1;
    bool flag = true; //是否随机成功
    while (flag)
    {
        flag = false;
        FOOD_X = random(COLUMN); //产生随机数[1,COLUMN]
        FOOD_Y = random(ROW);    //产生随机数[1,ROW]
        p = head->next;
        if (FOOD_X == head->x && FOOD_Y == head->y) //判断是否与蛇头重合
        {
            flag = true;
            continue;
        }
        while (p != NULL) //判断是否与蛇身重合
        {
            if (FOOD_X == p->x && FOOD_Y == p->y)
            {
                flag = true;
                break;
            }
            p = p->next;
        }
		/*p1=head;
        if(head->direction==RIGHT)
        {
			FOOD_X=(p1->x)%COLUMN+1;
        	FOOD_Y=p1->y;
    	}
    	if(head->direction==LEFT)
        {
			FOOD_X=(p1->x-2+COLUMN)%COLUMN+1;
        	FOOD_Y=p1->y;
    	}
    	if(head->direction==UP)
        {
			FOOD_X=p1->x;
       		FOOD_Y=(p1->y-2+ROW)%ROW+1;
    	}
    	if(head->direction==DOWN)
        {
			FOOD_X=p1->x;
        	FOOD_Y=(p1->y)%ROW+1;
    	}*/ 
    }
    position.X = 2 * FOOD_X - 1 + RAW_X; //设置光标位置
    position.Y = FOOD_Y + RAW_Y;
    SetConsoleCursorPosition(screen_out, position);
    printf("◆"); //输出食物
    FOOD = true; //当前食物存在
}


bool eatSnakeBody() //下一个是否为蛇身
{
    int next_x;
    int next_y;
    SnakeBody *p = head->next;
    switch (head->direction)
    {
    case UP:
        next_x = head->x;
        next_y = ((head->y - 2 + ROW) % ROW) + 1;
        break;
    case RIGHT:
        next_x = (head->x % COLUMN) + 1;
        next_y = head->y;
        break;
    case DOWN:
        next_x = head->x;
        next_y = (head->y % ROW) + 1;
        break;
    case LEFT:
        next_x = ((head->x - 2 + COLUMN) % COLUMN) + 1;
        next_y = head->y;
        break;
    }
    while (p != NULL)
    {
        if (p->x == next_x && p->y == next_y)
        {
            return true;
        }
        p = p->next;
    }
    return false;
}

bool eatWall() //是否撞墙
{
     int next_x;
     int next_y;
     SnakeBody *p = head->next;
     switch (head->direction)
     {
    case UP:
        next_x = head->x;
        next_y = head->y - 1;
        break;
    case RIGHT:
        next_x = head->x + 1;
        next_y = head->y;
        break;
    case DOWN:
        next_x = head->x;
        next_y = head->y + 1;
        break;
    case LEFT:
        next_x = head->x - 1;
        next_y = head->y;
        break;
     }
     if (head->direction == UP && next_y == 0)
     {
         return true;
     }
     if (head->direction == DOWN && next_y == ROW+1)
     {
         return true;
     }
     if (head->direction == LEFT && next_x == 0)
     {
         return true;
     }
     if (head->direction == RIGHT && next_x == COLUMN+1)
     {
         return true;
     } 
     return false;
}

bool eatFood() //下一个是否为食物
{
    int next_x;
    int next_y;
    SnakeBody *p = head->next;
    switch (head->direction)
    {
    case UP:
        next_x = head->x;
        next_y = ((head->y - 2 + ROW) % ROW) + 1;
        break;
    case RIGHT:
        next_x = (head->x % COLUMN) + 1;
        next_y = head->y;
        break;
    case DOWN:
        next_x = head->x;
        next_y = (head->y % ROW) + 1;
        break;
    case LEFT:
        next_x = ((head->x - 2 + COLUMN) % COLUMN) + 1;
        next_y = head->y;
        break;
    }
    if (next_x == FOOD_X && next_y == FOOD_Y)
    {
        return true;
    }
    return false;
}

void autoMoveSnake1() //普通移动
{
    position.X = 2 * rear->x - 1 + RAW_X;           //清除蛇尾
    position.Y = rear->y + RAW_Y;                   //
    SetConsoleCursorPosition(screen_out, position); //
    putchar(' ');                                   //
    putchar(' ');
    position.X = 2 * head->x - 1 + RAW_X;           //更改原蛇头符号
    position.Y = head->y + RAW_Y;                   //
    SetConsoleCursorPosition(screen_out, position); //
    printf("○");                                    //
    _moveSnake();                                   //移动
    position.X = 2 * head->x - 1 + RAW_X;           //输出新蛇头
    position.Y = head->y + RAW_Y;                   //
    SetConsoleCursorPosition(screen_out, position); //
    printf("◎");                                    //
    Sleep(head->speed);                             //控制速度
}

void autoMoveSnake2() //吃食物移动
{
    FOOD = false;                                   //食物置false
    head->score++;                                  //得分加一
    position.X = 2 * head->x - 1 + RAW_X;           //更改原蛇头符号
    position.Y = head->y + RAW_Y;                   //
    SetConsoleCursorPosition(screen_out, position); //
    printf("○");                                    //
    _lengthenSnake();                               //增加一截蛇身
    position.X = 2 * head->x - 1 + RAW_X;           //输出新蛇头
    position.Y = head->y + RAW_Y;                   //
    SetConsoleCursorPosition(screen_out, position); //
    printf("◎");                                    //
    Sleep(head->speed);                             //控制速度
}

void GameMode(int gamemodenumber) //开始游戏 
{
    char key;
    bool pause_game = false;
    FOOD = false; //初始没有食物
    while (true)
    {
        SetConsoleCursorInfo(screen_out, &cursor_info);
        if (!FOOD)
        {
            createFood();
        }
        if (_kbhit())
        {
            key = getch();
            if (key == 'q')
            {
                clearSnake();
                fflush(stdin);
                break;
            }
            else if (key == 'p') //暂停功能
            {
                position.X = 2 * COLUMN + 5 + RAW_X;
                position.Y = (int)ROW / 2 + 11 + RAW_Y;
                SetConsoleCursorPosition(screen_out, position);
                if (pause_game == true)
                {
                    printf("             ");
                    pause_game = false;
                }
                else
                {
                    printf("    暂停中...");
                    pause_game = true;
                }
                fflush(stdin);
            }
            else if (key == DIRECTIONKEY && !pause_game)
            {
                key = getch();
                changeDirection(key);
            }
            else
            {
                fflush(stdin);
            }
        }
        if (!pause_game)
        {        	
        	if(gamemodenumber==2)
        	{
				if(eatWall())
        		{
	        		Sleep(500);
	                clearSnake(50);
	                updateHighestScore(); //更新历史最高分
	                break;	
				}
			}
			
            if (eatSnakeBody())
            {
                Sleep(500);
                clearSnake(50);
                updateHighestScore(); //更新历史最高分
                break;
            }
            else if (eatFood())
            {
                autoMoveSnake2();
                updateScore();
                if (head->speed > 20) //吃一个食物就加速
                {
                    head->speed -= 3;
                }
            }
            else
            {
                autoMoveSnake1();
            }
        }
    }
}

void mainLoopSnake()
{
    system("chcp 936"); //保证不乱码
    system("color 0b"); //颜色
    char key;
    while (true)
    {
    	int gamemodenumber=selectgamemodenumber();
        initScreen();
        initSnake();
        printSnake();
        getch();
        GameMode(gamemodenumber);		//游戏开始 
        delSnake();
        position.X = (int)COLUMN * 4 / 5 - 2 + RAW_X;
        position.Y = (int)ROW / 2 - 1 + RAW_Y;
        SetConsoleCursorPosition(screen_out, position);
        printf("  游戏结束...");
        position.Y += 2;
        SetConsoleCursorPosition(screen_out, position);
        printf("'r'开始新的一局...");
        position.Y += 2;
        SetConsoleCursorPosition(screen_out, position);
        printf(" 'q'退出游戏...");
        key = '\0';
        while (key != 'r' && key != 'q')
        {
            key = getch();
        }
        if (key == 'q')
        {
            break;
        }
    }
    system("color 0f");
    system("cls");
}

int main()
{
    mainLoopSnake();
    return 0;
}
