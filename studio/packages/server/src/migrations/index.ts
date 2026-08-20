/**
 * 迁移管理中心
 *
 * 所有迁移文件在这里统一管理和导出
 * 添加新迁移时，按时间戳顺序添加到数组中
 */

import { MigrationInterface } from 'typeorm';
import { AddMessageReplyForeignKey1730000000000 } from './1730000000000-AddMessageReplyForeignKey';
import { MigrateSpanTable1740000000000 } from './1740000000000-MigrateSpanTable';
import { CreateManagementCacheTables1750000000000 } from './1750000000000-CreateManagementCacheTables';
import { AddResourceNodeId1760000000000 } from './1760000000000-AddResourceNodeId';

/**
 * 所有迁移列表（按时间戳顺序）
 */
import { AddRunNodeId1770000000000 } from './1770000000000-AddRunNodeId';
import { CreateChatSessionTable1780000000000 } from './1780000000000-CreateChatSessionTable';
import { AddUserGroupDefaultContainer1790000000000 } from './1790000000000-AddUserGroupDefaultContainer';
import { AddGroupContainerStatus1800000000000 } from './1800000000000-AddGroupContainerStatus';
import { CreateTrainingResourceTables1810000000000 } from './1810000000000-CreateTrainingResourceTables';
import { CreateResourceShareScopeTable1820000000000 } from './1820000000000-CreateResourceShareScopeTable';
import { CreateResourceAuditEventTable1830000000000 } from './1830000000000-CreateResourceAuditEventTable';
import { AddResourcePoolNodeAllowedGpuIndexes1840000000000 } from './1840000000000-AddResourcePoolNodeAllowedGpuIndexes';
import { AddAuthSessionLastSeenAt1850000000000 } from './1850000000000-AddAuthSessionLastSeenAt';
import { AddResourcePoolEnabled1860000000000 } from './1860000000000-AddResourcePoolEnabled';
import { AddAuthPasswordPolicy1870000000000 } from './1870000000000-AddAuthPasswordPolicy';;
import { AddTrainingReservationDiagnostics1880000000000 } from './1880000000000-AddTrainingReservationDiagnostics';
import { AddTrainingReservationTaskType1890000000000 } from './1890000000000-AddTrainingReservationTaskType';
import { AddUserGroupMultinodeContainer1900000000000 } from './1900000000000-AddUserGroupMultinodeContainer';
import { BackfillUserGroupGrpoContainer1910000000000 } from './1910000000000-BackfillUserGroupGrpoContainer';

/**
 * 所有迁移列表（按时间戳顺序）
 */
export const migrations: (new () => MigrationInterface)[] = [
    AddMessageReplyForeignKey1730000000000,
    MigrateSpanTable1740000000000,
    CreateManagementCacheTables1750000000000,
    AddResourceNodeId1760000000000,
    AddRunNodeId1770000000000,
    CreateChatSessionTable1780000000000,
    AddUserGroupDefaultContainer1790000000000,
    AddGroupContainerStatus1800000000000,
    CreateTrainingResourceTables1810000000000,
    CreateResourceShareScopeTable1820000000000,
    CreateResourceAuditEventTable1830000000000,
    AddResourcePoolNodeAllowedGpuIndexes1840000000000,
    AddAuthSessionLastSeenAt1850000000000,
    AddResourcePoolEnabled1860000000000,
    AddAuthPasswordPolicy1870000000000,
    AddTrainingReservationDiagnostics1880000000000,
    AddTrainingReservationTaskType1890000000000,
    AddUserGroupMultinodeContainer1900000000000,
    BackfillUserGroupGrpoContainer1910000000000,
];


