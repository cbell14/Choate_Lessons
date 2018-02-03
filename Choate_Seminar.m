%% Sine Wave Fun
% clear all; close all; fclose all;
% clc;
% 
% steps = 500;
% t = linspace(-pi,pi,steps)/pi;
% ex1 = sin(2*pi*t);
% ex2 = sin(2*2*pi*t);
% ex3 = sin(4*2*pi*t);
% ex4 = 2*sin(2*pi*t);
% 
% figure('Name','Sine Wave Frequency Example')
% plot(t,ex1,'k-','linewidth',3,'DisplayName','Baseline Frequency')
% hold on; grid on;
% plot(t,ex2,'b-','linewidth',3,'DisplayName','Doubled Frequency')
% plot(t,ex3,'r-','linewidth',3,'DisplayName','Quadrupled Frequency')
% h = legend('show','location','best');
% xlabel('x/\pi','FontSize',16)
% ylabel('Amplitude','Fontsize',16)
% title(get(gcf,'Name'),'FontSize',16)
% set(h,'FontSize',14)
% 
% figure('Name','Sine Wave Amplitude Example')
% plot(t,ex1,'k-','linewidth',3,'DisplayName','Baseline Amplitude')
% hold on; grid on;
% plot(t,ex4,'b-','linewidth',3,'DisplayName','Doubled Amplitude')
% h = legend('show','location','best');
% xlabel('x/\pi','FontSize',16)
% ylabel('Amplitude','Fontsize',16)
% title(get(gcf,'Name'),'FontSize',16)
% set(h,'FontSize',14)

%% Fourier Series Example
clear all; clc

% We are going to be calculating a few different Fourier Series with
% different number of terms. To do so first we must calculate a0, an, and
% bn for all the terms that we will be using

N = 40; %variable indicating maximum number of sine terms we will use in our summation
syms x;

an = zeros(N,1);
bn = zeros(N,1);

a0 = (1/(2*pi))*(exp(pi)-exp(-pi)); %exact solution for a0, need to add 1/2 too the approx for a0 in slides

%loop to solve for an and bn out to 40 terms
for n=1:1:N
    syms x
    
    %use matlab to integrate and solve for an and bn 
    %int(function,wrt x, from pi to pi
    an_sym = (1/pi)*int(exp(x)*cos(n*x),x,-pi,pi); %integral form of an
    bn_sym = (1/pi)*int(exp(x)*sin(n*x),x,-pi,pi); %integral form of bn
    
    an(n,1) = eval(an_sym); %evaluate the symbolic integral
    bn(n,1) = eval(bn_sym);
    
end

%Now we need to calculate the Fourier Series Approximations using the a0,
%an, and bn terms we have solved for

nx = 500; %number of x steps to solve the function at
x = linspace(-pi,pi,nx); %range
f_x_actual =  exp(x); % exact solution

%40 Terms
f_tot40 = zeros(1,nx);
for n=1:1:40
    f_x_approx(n,:) = an(n,1).*cos(n*x) + bn(n,1).*sin(n*x); %calculate the nth term
    f_tot40(1,:) = f_tot40(1,:) + f_x_approx(n,:); %add the nth term to the previously calculated terms
end
f_tot40 = f_tot40 + a0;

%20 Terms
f_tot20 = zeros(1,nx);
for n=1:1:20
    f_x_approx(n,:) = an(n,1).*cos(n*x) + bn(n,1).*sin(n*x);
    f_tot20(1,:) = f_tot20(1,:) + f_x_approx(n,:);
end
f_tot20(1,:) = f_tot20(1,:) + a0;

%10 Terms
f_tot10 = zeros(1,nx);
for n=1:1:10
    f_x_approx(n,:) = an(n,1).*cos(n*x) + bn(n,1).*sin(n*x);
    f_tot10(1,:) = f_tot10(1,:) + f_x_approx(n,:);
end
f_tot10(1,:) = f_tot10(1,:) + a0;

%5 Terms
f_tot5 = zeros(1,nx);
for n=1:1:5
    f_x_approx(n,:) = an(n,1).*cos(n*x) + bn(n,1).*sin(n*x);
    f_tot5(1,:) = f_tot5(1,:) + f_x_approx(n,:);
end
f_tot5(1,:) = f_tot5(1,:) + a0;

%2 Terms
f_tot2 = zeros(1,nx);
for n=1:1:2
    f_x_approx(n,:) = an(n,1).*cos(n*x) + bn(n,1).*sin(n*x);
    f_tot2(1,:) = f_tot2(1,:) + f_x_approx(n,:);
end
f_tot2(1,:) = f_tot2(1,:) + a0;

%Plot
figure('Name','Fourier Series Example')
plot(x/pi,f_x_actual,'k-','linewidth',3,'DisplayName','e^x');
hold on; grid on;
plot(x/pi,f_tot40,'r-','linewidth',3,'DisplayName','N=40');
plot(x/pi,f_tot20,'g-','linewidth',3,'DisplayName','N=20');
plot(x/pi,f_tot10,'b-','linewidth',3,'DisplayName','N=10');
plot(x/pi,f_tot5,'m-','linewidth',3,'DisplayName','N=5');
plot(x/pi,f_tot2,'c-','linewidth',3,'DisplayName','N=2');
h = legend('show','location','best');
xlabel('x/\pi','FontSize',16)
ylabel('Amplitude','Fontsize',16)
title(get(gcf,'Name'),'FontSize',16)
set(h,'FontSize',14)

%% FFT Example
% Fs = 1000;            % Sampling frequency, 1000 samples per second
% T = 1/Fs;             % Sampling period, length of time per sample
% L = 1500;             % Length of signal
% t = (0:L-1)*T;        % Time vector, make time steps at each frequency sampling time
% 
% % create a generic signal that is the sum of 2 sine waves with frequencies 
% % of 50 and 120 Hz and 0.7 and 1 for amplitudes respectively
% S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t); 
% 
% %Plot uncorrupted signal in the time domain
% figure
% plot(1000*t(1:50),S(1:50))
% title('Original Time Domain Signal')
% xlabel('t (milliseconds)')
% ylabel('X(t)')
% 
% %Take FFT of original signal
% %Compute the two-sided spectrum P2.
% Y = fft(S);
% P2 = abs(Y/L); %normalize by length of signal
% P1 = P2(1:L/2+1); %convert to a single sided spectrum
% % When converting to a single sided spectrum from a double sided spectrum we 
% % must multiple by 2. The easiest way to visualize this is to use the area under the curve.
% % A two sided spectrum is symmetric about the y axis and the area under the
% % curve is the power of the signal. To keep the same amount of power in a
% % single sided spectrum as the double sided spectrum we must multiply it by
% % 2.
% P1(2:end-1) = 2*P1(2:end-1); 
% 
% 
% %Plot FFT of original signal
% f = Fs*(0:(L/2))/L;
% figure
% plot(f,P1,'b-') 
% title('Single-Sided Spectrum of Original Signal')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% 
% % corrupt the signal with generic white noise
% X = S + 2*randn(size(t));
% 
% %Plot the corrupted signal, we can't see the underlying frequencies anymore
% figure
% plot(1000*t(1:50),X(1:50))
% title('Signal Corrupted with Zero-Mean Random Noise')
% xlabel('t (milliseconds)')
% ylabel('X(t)')
% 
% %Take FFT of corrupted signal
% Y = fft(X);
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% 
% %Plot FFT of corrupted signal
% figure
% plot(f,P1) 
% title('Single-Sided Spectrum of Corrupted Signal')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
