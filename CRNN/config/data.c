#include "reg52.h"			 //此文件中定义了单片机的一些特殊功能寄存器
#include <stdlib.h>
#include<math.h>
typedef unsigned long u32;	   //对数据类型进行声明定义
typedef unsigned int u16;	 
typedef unsigned char u8;
u8 buff[8];
sbit LSA=P2^2;
sbit LSB=P2^3;
sbit LSC=P2^4;

u8 code smgduan[17]={0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,
					0x7f,0x6f};//显示0~F的值

void delay(u16 i)
{
	while(i--);	
}

void update(u32 val)  //更新缓冲数组
{	
	//分离各位数据
	buff[0]=val/10000000%10;	//12345678/10000000%10=1;
	buff[1]=val/1000000%10;	 	//12345678/1000000%10=2;
	buff[2]=val/100000%10;		//12345678/100000%10=3;
	buff[3]=val/10000%10;		//12345678/10000%10=4;
	buff[4]=val/1000%10;     	//12345678/1000%10=5;
	buff[5]=val/100%10;	    	//12345678/100%10=6;
	buff[6]=val/10%10;	     	//12345678/10%10=7;
	buff[7]=val%10;		        //12345678%10=8;
}

void display()
{
	u8 i;
	for(i=0;i<8;i++)
	{
		switch(i)	 //位选，选择点亮的数码管，
		{
			case(0):
				LSA=0;LSB=0;LSC=0; break;//显示第0位
			case(1):
				LSA=1;LSB=0;LSC=0; break;//显示第1位
			case(2):
				LSA=0;LSB=1;LSC=0; break;//显示第2位
			case(3):
				LSA=1;LSB=1;LSC=0; break;//显示第3位
			case(4):
				LSA=0;LSB=0;LSC=1; break;//显示第4位
			case(5):
				LSA=1;LSB=0;LSC=1; break;//显示第5位
			case(6):
				LSA=0;LSB=1;LSC=1; break;//显示第6位
			case(7):
				LSA=1;LSB=1;LSC=1; break;//显示第7位	
		}
		P0=smgduan[buff[i]];//发送段码
		delay(100); //间隔一段时间扫描	
		P0=0x00;//消隐
	}
}
void main(void)
{
	u32 num = 3966;	 //要显示的数据
	u16 t;		 //延时用
	while(1)
	{
		update(num);	   //更新显示缓冲数组
		for(t=0; t<50; t++)	  //用for循环来延时不然显示不正常
		{
			display();		//显示缓冲数组
		}
		num+=1;	 //显示数据自加
		if(num==3968)
		   break;
	}
}