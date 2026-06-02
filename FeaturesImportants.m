    clear; 
    clc; 
    close all;
    
    %Configuració de la carpeta
    RUTA_CARPETA = "C:\Users\dddro\Desktop\DEFINITIVA\DEFINITIVA";     
    Fs = 1000;                  % Freqüència de mostreig [Hz]
    f0 = 25;                    % Freqüència fonamental aproximada [Hz]
    L = 1024;                   % Mida de la finestra
    overlap = 0.3;              % Solapament del 30%
    step = round(L*(1-overlap)); % Salt --> nombre sencer
    estats = ["S","U","L"];    % Sa, Unbalance, Looseness
    
    % VARIABLES DEL DATASET 
    X = [];
    ESTAT_CONCRET = strings(0,1);
    MOTOR_CONCRET = [];
    REPE_CONCRET = [];
    
    % LECTURA D'ARXIUS
    for e = 1:length(estats)
        ESTAT = estats(e);
        CARPETA_ESTAT = fullfile(RUTA_CARPETA, ESTAT);
        ARXIUS = dir(fullfile(CARPETA_ESTAT, "*.txt"));
        disp(CARPETA_ESTAT)
        disp(length(ARXIUS))
        
        for k = 1:length(ARXIUS)
            NOM = ARXIUS(k).name;
            RUTA = fullfile(ARXIUS(k).folder, NOM);
            
            % Llegir senyal crua
            x = read_txt_motor(RUTA);
            
            % Treure offset
            x = x - mean(x);
            
            % Interpretar el nom de l'arxiu 
            PARTS = split(erase(NOM, ".txt"), "_"); %Borrar el .txt i separar per _
            PART_ESTAT = string(PARTS(1));         %La primera part, l'estat, és una lletra (str)      
            MOTOR_NUM = str2double(erase(string(PARTS(2)), "M")); %La segona és el motor però se li ha de borrar la M
            REPE_NUM = str2double(PARTS(3));         %La última part és la repetició        
            
            % Aplicació de finestres
            for i = 1:step:(length(x)-L+1)
                finestra = x(i:i+L-1);
                finestra = finestra - mean(finestra);
                
                % EXTREURE FEATURES 
                features = features_motor(finestra, Fs, f0);
                
                % GUARDAR FILA AL DATASET
                X = [X; features];
                ESTAT_CONCRET(end+1,1) = PART_ESTAT;
                MOTOR_CONCRET(end+1,1) = MOTOR_NUM;
                REPE_CONCRET(end+1,1) = REPE_NUM;
                
            end
        end
    end
    
    % CREAR TAULA FINAL
    T = array2table(X, 'VariableNames', {'KURT', 'H12', 'Ratio41','Pct_E_40_160','Pct_E_160_500','SpectralCentroid', 'RMS', 'CrestFactor'});
    
    T.ESTAT_CONCRET = categorical(ESTAT_CONCRET);
    T.MOTOR_CONCRET = MOTOR_CONCRET;
    T.REPE_CONCRET = REPE_CONCRET;
    
    disp(head(T));
    
    % GUARDAR CSV
    writetable(T, "C:\Users\dddro\Desktop\MESURES\DATASET\features_importants.csv");
    
    % FUNCIONS
    function x = read_txt_motor(filename)
        txt = fileread(filename);
        ini = strfind(txt, 'INICI_TRANS');
        fi  = strfind(txt, 'FI_TRANS');
        if isempty(ini)
            error("No s'ha trobat la marca INICI_TRANS a: %s", filename);
        end
        BLOC = txt(ini(1):fi(1));
        VALORS = regexp(BLOC, '[-+]?\d+', 'match');
        x = str2double(VALORS(:));
    end
    
    function features = features_motor(x, Fs, f0)
        N = length(x);
        
        KURT = kurtosis(x);
    
        w = hann(N);
        xw = x .* w;
        X = abs(fft(xw))/N;
        X = 2*X(1:(N/2));
        f = linspace(0, Fs/2, length(X))';
    
        % Càlcul dels harmònics
        H12 = pic_aprox(f, X, 12*f0, 3);
        H4 = pic_aprox(f, X, 4*f0, 3);
        H1 = pic_aprox(f, X, f0, 3);
        H2 = pic_aprox(f, X, 2*f0, 3);
    
        % Ràtios 
        Ratio41 = H4 / (H1);
    
        %RMS i CrestFactor
        RMS = sqrt(mean(x.^2));
        CrestFactor = max(abs(x)) / RMS;
       
        % Energia bandes, sobre 1 (percentatge)
        Energia_Total = sum(X.^2);
        Pct_E_40_160  = energia_banda(f, X, 40, 160) / Energia_Total;
        Pct_E_160_500 = energia_banda(f, X, 160, 500) / Energia_Total;
        
        % Forma de l'espectre
        SpectralCentroid = sum(f .* X) / (sum(X));
        
        features = [ KURT H12 Ratio41 Pct_E_40_160 Pct_E_160_500 SpectralCentroid RMS CrestFactor];
    end
    
    function A = pic_aprox(f, X, f_obj, TOLERANCIA) %Funció per trobar els pics principals propers a 25Hz
        index = abs(f - f_obj) <= TOLERANCIA;
        A = max(X(index));
    end
    function E = energia_banda(f, X, f1, f2) %Funcio pels percentatges d'energia a les tres bandes
        E = sum(X(f >= f1 & f < f2).^2);
    end