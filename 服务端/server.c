#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>

#define BAUDRATE B115200 ///Baud rate : 115200
#define DEVICE "/dev/ttyS1"//设置端口号
#define FALSE 0
#define TRUE 1
#define _POSIX_SOURCE 1 //POSIX系统兼容
int SerialPort_Send(int i){

	int fd,res;
	struct termios oldtio,newtio;
	
	fd=open(DEVICE,O_RDWR | O_NOCTTY);
	if(fd<0){
		perror(DEVICE);
		exit(-1);
	}
	tcgetattr(fd,&oldtio);//保存原来的参数
	bzero(&newtio,sizeof(newtio));
	newtio.c_cflag=BAUDRATE | CS8 | CLOCAL | CREAD | HUPCL;
	newtio.c_iflag=IGNBRK;
	newtio.c_oflag=0;
	newtio.c_lflag=ICANON;
	tcflush(fd,TCIFLUSH);
	tcsetattr(fd,TCSANOW,&newtio);//设置串口参数
	printf("%d\n",i);
	if(i==0){
		char openbuf[255]={0xdd,0x05,0x24,0x00,0x09};
		char closebj[255]={0xdd,0x05,0x24,0x00,0x03};
		write(fd,openbuf,5);
		write(fd,closebj,5);
		close(fd);
	}
	else{
		char closebuf[255]={0xdd,0x05,0x24,0x00,0x0a};
		char baojing[255]={0xdd,0x05,0x24,0x00,0x02};
		write(fd,closebuf,5);
		write(fd,baojing,5);
		close(fd);
	}

}

int main()
{
	int sockfd, new_fd;
	struct sockaddr_in my_addr;
	struct sockaddr_in their_addr;
	int sin_size;
	//建立TCP套接口
	if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
	{
		printf("create socket error");
		perror("socket");
		exit(1);
	}
	//初始化结构体，并绑定6666端口
	my_addr.sin_family = AF_INET;
	my_addr.sin_port = htons(6666);
	my_addr.sin_addr.s_addr = INADDR_ANY;
	bzero(&(my_addr.sin_zero), 8);
	int on;
	on = 1;
	setsockopt( sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on) );
	//绑定套接口
	if (bind(sockfd, (struct sockaddr*)&my_addr, sizeof(struct sockaddr)) == -1)
	{
		perror("bind socket error");
		exit(1);
	}
	//创建监听套接口
	if (listen(sockfd, 10) == -1)
	{
		perror("listen");
		exit(1);
	}
	//等待连接
	while (1)
	{
		sin_size = sizeof(struct sockaddr_in);
		printf("server is run......\n");
		//如果建立连接，将产生一个全新的套接字
		if ((new_fd = accept(sockfd, (struct sockaddr*)&their_addr, &sin_size)) == -1)
		{
			perror("accept");
			exit(1);
		}
		printf("accept success.\n");
		//break;

		//生成一个子进程来完成和客户端的会话，父进程继续监听
		if (!fork())
		{
			printf("create new thred success.\n");
			//读取客户端发来的信息
			int numbytes;
			char buff[1024];
			memset(buff, 0, 1024);
			if ((numbytes = recv(new_fd, buff, sizeof(buff), 0)) == -1)
			{
				perror("recv");
				exit(1);
			}
			printf("%s\n", buff);
			printf("--------------------------------------------------------\n\n");
			int i=(strcmp(buff,"0"));
			SerialPort_Send(i);
			/*if(i==0)
			{	
				char success[]="success";
				if (send(new_fd, success, strlen(success), 0) == -1)
					perror("send");
			}
			else{
				char failed[]="failed";
				if (send(new_fd, failed, strlen(failed), 0) == -1)
					perror("send");
			}
			
			close(new_fd);
			exit(0);
			}*/
			close(new_fd);
		}
	}
	close(sockfd);
}
