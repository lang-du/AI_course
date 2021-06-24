% moea/d
% https://blog.csdn.net/qq_35414569/article/details/79655400

% ==========init=================
M = 2;  % 目标函数个数，必须为2
iter = 200;    % 终止条件,别太大,不然速度很慢，可以优化代码减少时间
N = 30; % 种群大小
% lambda_ = rand(N, M);   % 随机权重系数
lambda_ = [linspace(0, 1, N); 1-linspace(0, 1, N)]';    % 均匀
lambda_ = lambda_ ./ sum(lambda_, 2);    % 权重系数为M列，每行和为1
T = 5;  % 邻域权重向量的个数，必须小于N

% 计算最近的T个权重
lambda_dist = dist(lambda_, lambda_');
[~, I] = sort(lambda_dist, 2);
B = I(:, 1:T);

% 初始化种群
X_number = 2;   % 变量个数
p_num = 100;    % 粒子个数
X = rand(N, X_number);
[F1, F2] = zdt3(X);
FV = fitness(F1, F2, lambda_);
z = [min(F1) min(F2)];

% pbest
pbest = X;          % 随机初始化pbest，可以通过计算T个临近的粒子优化
[F1, F2] = zdt3(pbest);
pbestFitness = fitness(F1, F2, lambda_);
w = 0.4;            % 速度权重参数
posV = zeros(size(pbest));    % 初始化粒子速度

c1 = 1;   % 速度加权系数
c2 = 1;   % 速度加权系数

v_max = 0.2;  % 粒子最大速度
v_min = -0.2; % 粒子最小速度

p_max = 2;  % 粒子最大值
p_min = 0;  % 粒子最小值

EP = [];
for it=1:iter
    % =========================update==========================
    for i = 1:N
        % 基因重组
        rand_index = randi(T, 2,1); % 随机挑选序号

        % 获取xk和xl
        xk_index = B(i, rand_index(1));
        xl_index = B(i, rand_index(2));
        xk = X(xk_index, :);
        xl = X(xl_index, :);

        % 遗传产生y，并判断该粒子是否最优
        % r = rand();
        r = 0.5;
        y = r .* xk + (1 - r) .* xl;    % 下面伪代码中的Xi
        % 粒子群更新,跟新流程参考https://blog.csdn.net/daaikuaichuan/article/details/81382794
        %========找出T个临近粒子中最优的粒子gbest====
        gbest = X(i, :);
        [F1, F2] = zdt3(gbest);
        gbestFitness = fitness(F1, F2, lambda_(i, :));
        for t = 1:T
            [F1, F2] = zdt3(X(B(i, t), :));
            pFitness = fitness(F1, F2, lambda_(i, :));
            if gbestFitness > pFitness
                gbest = X(B(i, t), :);
                gbestFitness = pFitness;
            end
        end
        
        % =======pso更新y==============
        posV(i,:) = w * posV(i, :) + c1 * rand() * (pbest(i, :) - y) + c2 * rand() * (gbest  - y);
        
        posV(posV > v_max) = v_max; % 速度限制
        posV(posV < v_min) = v_min; % 速度限制

        y = y + posV(i, :);   % 更新粒子位置

        y(y > p_max) = p_max;
        y(y < p_min) = p_min;
        
        
        % ===================计算y适应度===============
        [F1, F2] = zdt3(y);
        yFitness = fitness(F1, F2, lambda_(i, :));
        if yFitness < pbestFitness(i, :)
            pbestFitness(i, :) = yFitness;
            pbest(i, :) = y;
        end
        

        y_ = y;
        % 更新z
        [F1, F2] = zdt3(y_);
        F_y_= [F1 F2];
        for j = 1:M
            if z(j) < F_y_(j)
                z(j) = F_y_(j);
            end
        end

        % 更新领域解
        for k = 1:T
            j = B(i, k);
            % y_
            temp_gte = lambda_(j, :) .* (F_y_ - z);
            gte_y_ = max(temp_gte);

            % xj
            xj = X(j,:);
            [F1, F2] = zdt3(xj);
            F_xj = [F1, F2];
            temp_gte = lambda_(j, :) .* (F_xj - z);
            gte_xj = max(temp_gte);

            if gte_y_ <= gte_xj
                X(j, :) = y_;

                [F1, F2] = zdt3(X(j, :));
                FV(j) = fitness(F1, F2, lambda_(j, :));
            end
        end

        % 更新EP
        EP = updateEP(EP, F_y_);
    end
end
size(EP)
plot(EP(:,1),EP(:,2),'*b');                 %作图
%axis([0,1,0,1]);
xlabel('F_1');ylabel('F_2');title('ZDT3')

