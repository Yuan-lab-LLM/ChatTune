import { BaseEntity, DataSource, ViewColumn, ViewEntity } from 'typeorm';

@ViewEntity({
    expression: (dataSource: DataSource) =>
        dataSource
            .createQueryBuilder()
            .select('COUNT(DISTINCT run.project)', 'totalProjects')
            .addSelect('COUNT(*)', 'totalRuns')
            // 一个月前的项目和运行数
            .addSelect(
                `COUNT(DISTINCT CASE 
            WHEN run.timestamp > strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 month') 
            THEN run.project END)`,
                'projectsMonthAgo',
            )
            .addSelect(
                `COUNT(CASE 
            WHEN run.timestamp > strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 month')
            THEN 1 END)`,
                'runsMonthAgo',
            )
            // 最近一周的项目和运行数
            .addSelect(
                `COUNT(DISTINCT CASE  
            WHEN run.timestamp > strftime('%Y-%m-%d %H:%M:%S', 'now', '-7 days')
            THEN run.project END)`,
                'projectsWeekAgo',
            )
            .addSelect(
                `COUNT(CASE 
            WHEN run.timestamp > strftime('%Y-%m-%d %H:%M:%S', 'now', '-7 days')
            THEN 1 END)`,
                'runsWeekAgo',
            )
            // 一年前的项目和运行数
            .addSelect(
                `COUNT(DISTINCT CASE 
            WHEN run.timestamp > strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 year')
            THEN run.project END)`,
                'projectsYearAgo',
            )
            .addSelect(
                `COUNT(CASE 
            WHEN run.timestamp > strftime('%Y-%m-%d %H:%M:%S', 'now', '-1 year')
            THEN 1 END)`,
                'runsYearAgo',
            )
            .addSelect(
                `(
            WITH RECURSIVE
            months(date) AS (
                SELECT date('now', 'start of month', '-11 months')
                UNION ALL
                SELECT date(date, '+1 month')
                FROM months
                WHERE date < date('now', 'start of month')
            ),
            monthly_counts AS (
                SELECT 
                    strftime('%Y-%m', months.date) as month,
                    COUNT(CASE 
                        WHEN strftime('%Y-%m', run.timestamp) = strftime('%Y-%m', months.date) 
                        THEN 1 
                    END) as count
                FROM months
                LEFT JOIN run_table run ON strftime('%Y-%m', run.timestamp) = strftime('%Y-%m', months.date)
                GROUP BY strftime('%Y-%m', months.date)
                ORDER BY month DESC
            )
            SELECT json_group_array(
                json_object(
                    'month', month,
                    'count', count
                )
            )
            FROM monthly_counts
        )`,
                'monthlyRuns',
            )
            .from('run_table', 'run'),
})
export class RunView extends BaseEntity {
    @ViewColumn()
    totalProjects: number;

    @ViewColumn()
    totalRuns: number;

    @ViewColumn()
    projectsWeekAgo: number;

    @ViewColumn()
    runsWeekAgo: number;

    @ViewColumn()
    projectsMonthAgo: number;

    @ViewColumn()
    runsMonthAgo: number;

    @ViewColumn()
    projectsYearAgo: number;

    @ViewColumn()
    runsYearAgo: number;

    @ViewColumn()
    monthlyRuns: string;
}
