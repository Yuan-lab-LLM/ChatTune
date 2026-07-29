import { DataSource, DataSourceOptions } from 'typeorm';
import { AuthDao } from './dao/Auth';
import { InputRequestDao } from './dao/InputRequest';
import { RunDao } from './dao/Run';
import { migrations } from './migrations';
import { AuthLoginFailureTable, AuthSessionTable, AuthUserTable } from './models/Auth';
import { ChatSessionTable } from './models/ChatSession';
import { InputRequestTable } from './models/InputRequest';
import { MessageTable } from './models/Message';
import { ModelInvocationView } from './models/ModelInvocationView';
import { ManagementCacheTable } from './models/ManagementCache';
import { RefreshJobTable } from './models/RefreshJob';
import { ReplyTable } from './models/Reply';
import { RunTable } from './models/Run';
import { RunView } from './models/RunView';
import { SpanTable } from './models/Trace';
import {
    GroupResourceQuotaTable,
    ResourcePoolNodeTable,
    ResourcePoolTable,
    TrainingReservationNodeTable,
    TrainingReservationTable,
    TrainingResourceLockTable,
} from './models/TrainingResource';
import {
    GroupNodeAssignmentTable,
    ResourceCatalogTable,
    ResourceAuditEventTable,
    ResourcePublicationRequestTable,
    ResourceShareScopeTable,
    UserGroupMemberTable,
    UserGroupTable,
} from './models/ResourceAccess';
import { ResourceAccessService } from './services/resourceAccessService';

export const initializeDatabase = async (
    databaseConfig: DataSourceOptions,
): Promise<void> => {
    try {
        const options = {
            ...databaseConfig,
            entities: [
                RunTable,
                RunView,
                MessageTable,
                ReplyTable,
                InputRequestTable,
                SpanTable,
                ModelInvocationView,
                ManagementCacheTable,
                RefreshJobTable,
                AuthUserTable,
                AuthSessionTable,
                AuthLoginFailureTable,
                ChatSessionTable,
                UserGroupTable,
                UserGroupMemberTable,
                GroupNodeAssignmentTable,
                ResourceCatalogTable,
                ResourceAuditEventTable,
                ResourcePublicationRequestTable,
                ResourceShareScopeTable,
                ResourcePoolTable,
                ResourcePoolNodeTable,
                GroupResourceQuotaTable,
                TrainingReservationTable,
                TrainingReservationNodeTable,
                TrainingResourceLockTable,
            ],
            synchronize: false,
            migrations: migrations,
            migrationsRun: false,
            logging: false,
        };

        const dataSource = new DataSource(options);
        await dataSource.initialize();
        await dataSource.runMigrations();

        if (databaseConfig.type === 'better-sqlite3') {
            await dataSource.query('DROP VIEW IF EXISTS run_view');
            await dataSource.query('DROP VIEW IF EXISTS model_invocation_view');
        }

        await dataSource.synchronize();
        if (databaseConfig.type === 'better-sqlite3') {
            const staleLockCount = await TrainingResourceLockTable.count();
            if (staleLockCount) {
                await TrainingResourceLockTable.clear();
                console.warn(`Cleared ${staleLockCount} stale training resource lock(s) after Studio startup`);
            }
        }

        const printingOptions = {
            ...options,
            entities: undefined,
            migrations: undefined,
        };
        console.debug(
            `Database initialized with options: ${JSON.stringify(printingOptions, null, 2)}`,
        );
        console.debug('Refresh the database ...');
        await AuthDao.ensureDefaultAdmin();
        await ResourceAccessService.ensureDefaultGroup();
        await RunDao.updateRunStatusAtBeginning();
        await InputRequestDao.updateInputRequests();
        console.debug('Done');
    } catch (error) {
        console.error('Error initializing database', error);
        throw error;
    }
};

