function bad_seed_ode_test()
    % Define the system of ODEs
    % degen eigenvalue direction (they collapse)
    stab_eps = 0.01;
    function dydt = odefun(t, y)
        a = y(1);
        b = y(2);
        c = y(3);
%          dydt = [
%              -1.379*a - 2.139*b - (0.793)*c + 0.609*a^2 + 0.616*a*b - 0.543*b^2 - 0.814*b*c - 0.246*c^2-stab_eps*a^3;
%               1.334*a + 2.058*b + 0.763*c - 0.559*a^2 - 0.544*a*b + 0.533*b^2 + 0.765*b*c + 0.231*c^2-stab_eps*b^3;
%              -1.193*a - 1.827*b - 0.676*c + 0.023*a^2 - 0.473*a*b - 0.773*b^2 - 0.290*b*c-stab_eps*c^3;
%          ];
          dydt = [
              -1.379*a - 2.139*b - 0.793*c - stab_eps*a^3;
               1.334*a + 2.058*b + 0.763*c - stab_eps*b^3;
              -1.193*a - 1.827*b - 0.676*c - stab_eps*c^3;
          ];
%           dydt = [
%               -1.379*a - 2.139*b - 0.793*c - stab_eps*a^3;
%                1.334*a + 2.058*b + 0.763*c - stab_eps*b^3;
%               -1.193*a - 1.827*b - 0.676*c - stab_eps*c^3;
%             ];
    end

    % Define the time span
    tspan = [0 494];

    % Define the initial conditions
%      y0 = [0.70361549 -0.77091281 0.87890854];
     y0 = [0.1 0.1 0.1];

    % Solve the system of ODEs using ode45
    [t, y] = ode45(@odefun, tspan, y0);

    % Plot the solutions
    figure;
    plot(t, y(:, 1), 'r', t, y(:, 2), 'g', t, y(:, 3), 'b');
    xlabel('Time');
    ylabel('Variables');
    legend('a', 'b', 'c');
    title('Solution of the System of ODEs');
    grid on;
end
