#include "reg52.h"			 //���ļ��ж����˵�Ƭ����һЩ���⹦�ܼĴ���
#include <stdlib.h>
#include<math.h>
typedef unsigned long u32;	   //���������ͽ�����������
typedef unsigned int u16;	 
typedef unsigned char u8;
u8 buff[8];
sbit LSA=P2^2;
sbit LSB=P2^3;
sbit LSC=P2^4;

u8 code smgduan[17]={0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,
					0x7f,0x6f};//��ʾ0~F��ֵ

void delay(u16 i)
{
	while(i--);	
}

void update(u32 val)  //���»�������
{	
	//�����λ����
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
		switch(i)	 //λѡ��ѡ�����������ܣ�
		{
			case(0):
				LSA=0;LSB=0;LSC=0; break;//��ʾ��0λ
			case(1):
				LSA=1;LSB=0;LSC=0; break;//��ʾ��1λ
			case(2):
				LSA=0;LSB=1;LSC=0; break;//��ʾ��2λ
			case(3):
				LSA=1;LSB=1;LSC=0; break;//��ʾ��3λ
			case(4):
				LSA=0;LSB=0;LSC=1; break;//��ʾ��4λ
			case(5):
				LSA=1;LSB=0;LSC=1; break;//��ʾ��5λ
			case(6):
				LSA=0;LSB=1;LSC=1; break;//��ʾ��6λ
			case(7):
				LSA=1;LSB=1;LSC=1; break;//��ʾ��7λ	
		}
		P0=smgduan[buff[i]];//���Ͷ���
		delay(100); //���һ��ʱ��ɨ��	
		P0=0x00;//����
	}
}
void main(void)
{
	u32 num = 3966;	 //Ҫ��ʾ������
	u16 t;		 //��ʱ��
	while(1)
	{
		update(num);	   //������ʾ��������
		for(t=0; t<50; t++)	  //��forѭ������ʱ��Ȼ��ʾ������
		{
			display();		//��ʾ��������
		}
		num+=1;	 //��ʾ�����Լ�
		if(num==3968)
		   break;
	}
}