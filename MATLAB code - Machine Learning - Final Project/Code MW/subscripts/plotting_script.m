%% Figure 1
if options.Figure1 == 1
    external_gaps_annual = xlsread('dataset.xlsx','external_gaps','B2:D23');
    UCM= xlsread('dataset.xlsx','external_gaps','G2:G93');

    [hptrend,hpcycle]=hpfilter(100*log(rawdata.nonFRED(2:end,1)),1600);

    figure; fig = gcf; fig.PaperOrientation = 'landscape';
    set(fig,'PaperUnits','normalized');
    set(fig,'PaperPosition', [0 0 1 1]);

    subplot(2,1,1)
    CEPRbc(dates, Baseline,{'-'},3,{'r'});
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    set(gca,'FontSize',14)
    title('Estimated BN Output Gap')

    subplot(2,2,3)
    h1 = CEPRbc(dates, Baseline,{'-'},3,{'r'});
    hold on
    h2=plot(1999.5:2019.5, external_gaps_annual(:,1),'-.b','LineWidth',2);
    hold on
    h3=plot(1999.5:2019.5, external_gaps_annual(:,2),'--k','LineWidth',2);
    hold on
    h4=plot(1999.5:2019.5, external_gaps_annual(:,3),'-*g','LineWidth',2,'MarkerSize',5);
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    ylim([-14  6]);
    title('Comparison Against External Estimates')
    set(gca,'FontSize',14)
    legend([h1 h2 h3 h4],{'Estimated BN Output Gap','OECD','European Commission','IMF WEO'},'Location','southwest')
 
    subplot(2,2,4)
    h1 = CEPRbc(dates, Baseline,{'-'},3,{'r'});
    hold on
    h2= plot(dates, hpcycle,'-.b','LineWidth',2);
    hold on
    h3=plot(dates,100*UCM,'--k','LineWidth',2);
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    ylim([-14  6]);
    title('Comparison Against Other Estimates')
    set(gca,'FontSize',14)
    legend([h1 h2 h3],{'Estimated BN Output Gap','HP Filter','UCM'},'Location','southwest')

    print( [ pwd '\figures\Figure1' ] , '-dpdf' );

end

%% Figure 2
if options.Figure2 == 1
    figure; fig = gcf; fig.PaperOrientation = 'landscape';
    set(fig,'PaperUnits','normalized');
    set(fig,'PaperPosition', [0 0 1 1]);

    subplot(2,1,1)
    h1 = CEPRbc(dates, Baseline,{'-'},3,{'r'});
    hold on
    h2=plot(dates, Gap_compare_other_covid_adjustment(:,2),'-.b','LineWidth',3);
    hold on
    h3=plot(dates,Gap_compare_other_covid_adjustment(:,1),'-*k','LineWidth',3,'MarkerSize',5);
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    set(gca,'FontSize',16)
    legend([h1 h2 h3],{'Baseline (Two-step)','Estimated full sample without correction','Estimated up to 2019Q4'},'Location','south')
    ylim([-18 8])

    subplot(2,1,2)
    h1 = CEPRbc(dates, Baseline,{'-'},3,{'r'});
    hold on
    h2=plot(dates, Gap_MH_Correction,'-.k','LineWidth',3);
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    set(gca,'FontSize',16)
    legend([h1 h2],{'Baseline (Two-step)','Bayes Correction'},'Location','south')
    ylim([-18 8])
    print( [ pwd '\figures\Figure2'] , '-dpdf' );

end

%% Figure 3
if options.Figure3 == 1

    figure; fig = gcf; fig.PaperOrientation = 'landscape';
    set(fig,'PaperUnits','normalized');
    set(fig,'PaperPosition', [0 0 1 1]);

    subplot(2,1,1)
    h1 = CEPRbc(dates,Baseline,{'-'},3,{'r'});
    hold on
    h2 = plot(dates,Gap_point075,'--k','LineWidth',2);
    hold on
    h3 = plot(dates,Gap_point2,'-.b','LineWidth',2,'MarkerSize',5);
    hold on
    h4 = plot(dates,Gap_point9,'LineWidth',2);
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    legend([h1 h2 h3 h4],...
        {'\lambda = 0.2 (Baseline)','\lambda = 0.075','\lambda = 0.75','\lambda = 0.9' },...
        'Location','south','Orientation','horizontal');
    set(gca,'FontSize',14)
    ylim([-15 7])

    subplot(2,1,2)
    h3=CEPRbc(dates,100*(rawdata.nonFRED(2:end,1)./repmat(rawdata.nonFRED(1,1),T,1)),{'-.'},3,{'k'});
    hold on
    h1 = plot(dates,100*(exp(log(rawdata.nonFRED(2:end,1))- Baseline/100))./repmat(rawdata.nonFRED(1,1),T,1),'-r','LineWidth',2);
    hold on
    h2 = plot(dates,100*(exp(log(rawdata.nonFRED(2:end,1))- Gap_point2/100))./repmat(rawdata.nonFRED(1,1),T,1),'--*b','LineWidth',2,'MarkerSize',2);
    ylim([98 135])
    set(gca,'FontSize',14)
    legend([h1 h2 h3],{'Trend Output, \lambda = 0.2 (Baseline)','Trend Output, \lambda = 0.75','Real GDP'},'Location','south','Orientation','horizontal')

    print( [ pwd '\figures\Figure3' ] , '-dpdf' );

end

%% Figure 4

if options.Figure4 == 1
    figure; fig = gcf; fig.PaperOrientation = 'landscape';
    set(fig,'PaperUnits','normalized');
    set(fig,'PaperPosition', [0 0 1 1]);

    subplot(2,1,1)
    bar(std(Info_decom_baseline.cycle(:,1:round(N/2))))
    set(gca, 'XTickLabel',Series(1:round(N/2)), 'XTick',1:1:round(N/2),'Fontsize',10)

    subplot(2,1,2)
    bar(std(Info_decom_baseline.cycle(:,round(N/2)+1:N)))
    set(gca, 'XTickLabel',Series(round(N/2)+1:N),'Fontsize',10)

    print( [ pwd '\figures\Figure4' ] , '-dpdf' );

end

%% Figure 5 Robustness Model Size
if options.Figure5 == 1
    figure; fig = gcf; fig.PaperOrientation = 'landscape';
    set(fig,'PaperUnits','normalized');
    set(fig,'PaperPosition', [0 0 1 1]);

    subplot(2,1,1)
    h1 = CEPRbc(dates, Baseline,{'-'},3,{'r'});
    hold on
    h2=plot(dates,univariate_BN_cycle,'-.b','LineWidth',2);
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    legend([h1 h2],{'Baseline','Univariate AR(4)'},'Orientation','Horizontal','Location','South')
    set(gca,'FontSize',14)
    title('Comparison of Baseline Relative to Output Gap from AR(4)')

    subplot(2,1,2)
    h1 = CEPRbc(dates, Baseline,{'-'},3,{'r'});
    hold on
    h2=plot(dates,Gap_alt1,'-*b','LineWidth',2);
    hold on
    h3=plot(dates,Gap_alt2,'--k','LineWidth',2);
    hold on
    h4=plot(dates,Gap_alt3,'-.g','LineWidth',2);
    plot([dates(1) dates(end)],zeros(2,1),'-k')
    legend([h1 h2 h3 h4],{'Baseline','12 variable','Drop Term Spread','Drop Real Exchange Rate'},'Orientation','Horizontal','Location','South')
    set(gca,'FontSize',14)
    title('Estimated Multivariate BN Output Gap')

    print( [ pwd '\figures\Figure5' ] , '-dpdf' );
end
%% Figure 6 hours decom

if options.Figure6 == 1

    figure; fig = gcf; fig.PaperOrientation = 'landscape';
    set(fig,'PaperUnits','normalized');
    set(fig,'PaperPosition', [0 0 1 1]);

    subplot(2,1,1)
    CEPRbc(dates,sum(Info_decom_baseline.cycle,2),{'-'},3,{'r'});
    hold on
    h2=bar(dates,[Info_decom_other_baseline Info_decom_baseline.cycle(:,9)],'stacked');
    h2(1,1).FaceColor = [1 1 1];
    h2(1,2).FaceColor = [0 0 0.3];
    legend(h2,{'Other Variables','Hours'},'Location','eastoutside')
    set(gca,'FontSize',14)
    ylim([-14 6])
    title('Baseline, \lambda = 0.2')

    subplot(2,1,2)
    CEPRbc(dates,sum(Info_decom_point2.cycle,2),{'-'},3,{'r'});
    hold on
    h2=bar(dates,[Info_decom_other_point2 Info_decom_point2.cycle(:,9)],'stacked');
    h2(1,1).FaceColor = [1 1 1];
    h2(1,2).FaceColor = [0 0 0.3];
    legend(h2,{'Other Variables','Hours'},'Location','eastoutside')
    set(gca,'FontSize',14)
    ylim([-14 6])
    title('\lambda = 0.75')

    print( [ pwd '\figures\Figure6' ] , '-dpdf' );

end

%% Figure 7 Hours unemployment plots
if options.Figure7 == 1

    figure; fig = gcf; fig.PaperOrientation = 'landscape';
    set(fig,'PaperUnits','normalized');
    set(fig,'PaperPosition', [0 0 1 1]);

    subplot(2,2,1)
    CEPRbc(dates,exp(y(:,9)/100),{'-'},3,{'r'});
    set(gca,'FontSize',14)
    title('Hours Worked')
    ylim([85 110])

    subplot(2,2,2)
    CEPRbc(dates,y(:,12),{'-'},3,{'r'});
    set(gca,'FontSize',14)
    title('Unemployment Rate')

    subplot(2,2,3)
    CEPRbc(dates,y(:,4),{'-'},3,{'r'});
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k','Linewidth',2);
    set(gca,'FontSize',14)
    title('Industrial Production Growth')

    subplot(2,2,4)
    CEPRbc(dates,y(:,11),{'-'},3,{'r'});
    set(gca,'FontSize',14)
    title('Capacity Utilization')
    ylim([65 90])

    print( [ pwd '\figures\Figure7' ] , '-dpdf' );

end


%% Figure 8

if options.Figure8 == 1

    figure; fig = gcf; fig.PaperOrientation = 'landscape';
    set(fig,'PaperUnits','normalized');
    set(fig,'PaperPosition', [0 0 1 1]);

    subplot(2,1,1)
    h1 = CEPRbc(dates,Baseline,{'-'},3,{'r'});
    hold on
    h2=plot(dates,Gap_no_block_exo,'-.k','LineWidth',3,'MarkerSize',5);
    hold on
    plot([dates(1) dates(end)],zeros(2,1),'-k');
    ylim([-25  10]);
    title('Estimated with two-step adjustment, \lambda = 0.75')
    set(gca,'FontSize',16)
    legend([h1 h2],{'Baseline','No Block Exogeneity'},'Location','south')

    subplot(2,1,2)
    CEPRbc(dates,Baseline,{'-'},3,{'r'});
    hold on
    h2=bar(dates,[sum(shock_decom_baseline.cycle(:,1:nBlockExo),2) sum(shock_decom_baseline.cycle(:,nBlockExo+1:end),2)],'stacked');
    cmap = colormap(jet);
    h2(1,1).FaceColor = [1 0 0];cmap(1,:);
    h2(1,2).FaceColor = [1 1 1];cmap(end,:);
    legend(h2,{'Foreign','Domestic'},'Location','south')
    set(gca,'FontSize',16)
    title('Shock Decomposition')

    print( [ pwd '\figures\Figure8' ] , '-dpdf' );

end