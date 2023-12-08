function bad_seed_ode_test()
    % Define the system of ODEs
    % Working with 
    % degen eigenvalue direction (they collapse)

% Parameters producing results for old version of SINDy
%     pickle_path = "plume_videos/July_20/video_low_2/gauss_time_blur/gauss_blur_coeff.pkl"
% 
%     with open(pickle_path, 'rb') as f:
%         loaded_arrays = pickle.load(f)
%     
%     # Params
%     time_series = loaded_arrays["mean"]
%     window_length = 5
%     ensem_thresh = 0.1
%     ensem_alpha=1e-3
%     ensem_max_iter=200
%     poly_degree=2
%     ensem_num_models=20
%     ensem_time_points = int(len(time_series)*0.6) # ensem_time_points = 100
%     seed =1
%     
    stab_eps = 0.01;
    perturb_eps = 0.00;
    function dydt = odefun(t, y)
        a = y(1);
        b = y(2);
        c = y(3);
         dydt = [
             -1.379*a - 2.139*b - (0.793)*c + 0.609*a^2 + 0.616*a*b - 0.543*b^2 - 0.814*b*c - 0.246*c^2-stab_eps*a^3;
              1.334*a + 2.058*b + 0.763*c - 0.559*a^2 - 0.544*a*b + 0.533*b^2 + 0.765*b*c + 0.231*c^2-stab_eps*b^3;
             -1.193*a - 1.827*b - 0.676*c + 0.023*a^2 - 0.473*a*b - 0.773*b^2 - 0.290*b*c-stab_eps*c^3;
         ];
%         dydt = [
%             -1.379*a - 2.139*b - (0.793+perturb_eps)*c - stab_eps*a^3;
%              1.334*a + 2.058*b + 0.763*c - stab_eps*b^3;
%             -1.193*a - 1.827*b - 0.676*c - stab_eps*c^3;
%         ];
    end

    % Define the time span
    tspan = [0 494+200];

    % Define the initial conditions
     y0 = [0.70361549 -0.77091281 0.87890854];
%      y0 = [0.1 0.1 0.1];

    % Solve the system of ODEs using ode45
    [t, y] = ode45(@odefun, tspan, y0);

    % Plot the solutions
    figure;
    plot(t, y(:, 1), 'r', t, y(:, 2), 'g', t, y(:, 3), 'b');
    xlabel('Time');
    ylabel('Variables');
    legend('a', 'b', 'c');
    title("Solution of the System of ODEs: eps="+stab_eps+", Peturb eps="+perturb_eps+"");
    grid on;

    str = "test " + stab_eps + "";
    disp(str)
    %% Looking at eigenvectors of linear system
    A = [-1.379 -(2.139) -(0.793+perturb_eps);
          1.334  2.058  0.763;
         -1.193 -1.827 -0.676];
    [V,D] = eig(A);
    disp(V);
    v1  = transpose(real(V(:,1)));
    v2 = transpose(real(V(:,2)));
    v3  = transpose(real(V(:,3)));
    origin = [0, 0, 0];

    figure;hold on;
    plot3([origin(1) v1(1)],[origin(2) v1(2)],[origin(3) v1(3)],'r-^', 'LineWidth',3);
    plot3([origin(1) v2(1)],[origin(2) v2(2)],[origin(3) v2(3)],'g-^', 'LineWidth',3);
    plot3([origin(1) v3(1)],[origin(2) v3(2)],[origin(3) v3(3)],'b-^', 'LineWidth',3);
    title("Eigenvectors of Linear ODE sys: eps="+stab_eps+", "+"perturb eps="+perturb_eps+"");
    grid on;
    xlabel('X axis'), ylabel('Y axis'), zlabel('Z axis')
    set(gca,'CameraPosition',[1 2 3]);
end