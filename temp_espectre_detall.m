fitxer = "L_M5_30.txt"; %canviar nom del fitxer

raw = readmatrix(fitxer); %es llegeix com a matriu numèrica
raw = raw(~isnan(raw)); %les linies de text es tornen buides
raw = raw - mean(raw); %es centra en 0

Fs = 1000; %freq de mostreig
T = 1/Fs; %període de mostreig
N = length(raw); %nombre total de mostres en la mesura

Eix_temps = linspace(0, (N-1)*T, N); %eix temporal

deltaf = Fs/N; %resolució en l'espectre de freqüències
Eix_freq = linspace(0, Fs-deltaf, N); %eix freqüencial

X = abs(fft(raw)); %fft de les dades
X = X/max(X); %normalitzar espectre respecte 1

figure;
subplot(2,1,1)
plot(Eix_temps, raw)
title('Senyal temporal')
xlabel('Temps [s]')
ylabel('Amplitud')
grid on

subplot(2,1,2)
plot(Eix_freq, X)
title('FFT')
xlabel('Freqüència [Hz]')
ylabel('Amplitud normalitzada')
xlim([0 500])
grid on