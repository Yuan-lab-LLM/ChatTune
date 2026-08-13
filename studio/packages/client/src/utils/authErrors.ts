type Translate = (key: string, options?: Record<string, unknown>) => string;

const authErrorMessages: Record<string, string> = {
    'auth.error.accountDisabled': 'auth.accountDisabled',
    'auth.error.invalidCredentials': 'auth.loginFailed',
    'auth.error.tooManyLoginAttempts': 'auth.tooManyLoginAttempts',
    'auth.error.usernameRequired': 'auth.usernameRequired',
    'auth.error.passwordRequired': 'auth.passwordRequired',
    'auth.error.currentPasswordRequired': 'auth.currentPasswordRequiredMessage',
    'auth.error.usernameTooShort': 'auth.usernameTooShort',
    'auth.error.usernameTooLong': 'auth.usernameTooLong',
    'auth.error.usernameInvalid': 'auth.usernameInvalid',
    'auth.error.passwordTooShort': 'auth.passwordTooShort',
    'auth.error.newPasswordTooShort': 'auth.newPasswordTooShort',
    'auth.error.usernameExists': 'auth.usernameExists',
    'auth.error.currentPasswordIncorrect': 'auth.currentPasswordIncorrect',
    'auth.error.userNotFound': 'auth.userNotFound',
    'auth.error.cannotModifySelf': 'auth.cannotModifySelf',
    'auth.error.cannotRemoveLastAdmin': 'auth.cannotRemoveLastAdmin',
    'auth.error.useLogoutForSelf': 'auth.useLogoutForSelf',
    'auth.error.adminRequired': 'auth.adminOnlyAction',
    'auth.error.loginRequired': 'auth.loginRequired',
    'auth.error.passwordChangeRequired': 'auth.passwordChangeRequired',
    'auth.error.groupRequired': 'auth.groupRequired',
    'auth.error.groupNotFound': 'auth.groupNotFound',
    'auth.error.defaultGroupCannotDelete': 'auth.defaultGroupCannotDelete',
    'auth.error.groupNotEmpty': 'auth.groupNotEmpty',
    'auth.error.groupNodeAssigned': 'auth.groupNodeAssigned',
    'auth.error.onlyUsersInGroups': 'auth.onlyUsersInGroups',
    'auth.error.resourceForbidden': 'auth.resourceForbidden',
    'auth.error.containerRolesMustDiffer': 'resourceAccess.containerRolesMustDiffer',
    'auth.error.containerNameInvalid': 'resourceAccess.containerNameInvalid',
    'auth.error.containerAlreadyBound': 'resourceAccess.containerAlreadyBound',
    'auth.error.containerNotFound': 'resourceAccess.containerNotFound',
    'auth.error.containerNotRunning': 'resourceAccess.containerNotRunning',
    'auth.error.containerValidationFailed': 'resourceAccess.containerValidationFailed',
    'auth.error.groupQuotaAssigned': 'resourceAccess.groupQuotaAssigned',
    'auth.error.groupResourcesShared': 'resourceAccess.groupResourcesShared',
    'auth.error.activeTrainingReservationExists': 'resourceAccess.activeTrainingReservationExists',
    'auth.error.groupQuotaAssignedBeforeNodeChange': 'resourceAccess.groupQuotaAssignedBeforeNodeChange',
    'auth.error.groupNodeRequired': 'resourceAccess.groupNodeRequired',
    'auth.error.resourceNotFound': 'resourceAccess.resourceNotFound',
    'auth.error.publicationRequestNotFound': 'resourceAccess.publicationRequestNotFound',
    '训练 Docker 与评测/推理 Docker 不能相同': 'resourceAccess.containerRolesMustDiffer',
    'Docker 容器名只能包含字母、数字、点、下划线和连字符': 'resourceAccess.containerNameInvalid',
    '该 Docker 容器已绑定其他用户组': 'resourceAccess.containerAlreadyBound',
    '请先删除用户组训练资源配额': 'resourceAccess.groupQuotaAssigned',
    '请先移除共享给该用户组的资源': 'resourceAccess.groupResourcesShared',
    '存在活跃训练预约，不能修改归属 Runtime 节点': 'resourceAccess.activeTrainingReservationExists',
    '请先删除用户组训练资源配额，再修改归属 Runtime 节点': 'resourceAccess.groupQuotaAssignedBeforeNodeChange',
    '请先为用户组分配资源节点': 'resourceAccess.groupNodeRequired',
    '资源不存在': 'resourceAccess.resourceNotFound',
    '发布申请不存在或已处理': 'resourceAccess.publicationRequestNotFound',
    '存在活跃训练预约，当前配置暂不可修改': 'trainingResource.activeReservationsBlockConfig',
    '资源池不存在': 'trainingResource.poolNotFound',
    '资源池已禁用': 'trainingResource.poolDisabled',
    '用户组没有已启用的训练资源池': 'trainingResource.noEnabledPool',
    '用户组不存在': 'auth.groupNotFound',
    '归属 Runtime 节点必须属于资源池': 'trainingResource.homeNodeMustBelongToPool',
    '资源池归属节点必须与用户组当前 Runtime 节点一致': 'trainingResource.homeNodeMustMatchGroupRuntime',
    '保底 GPU 数必须是非负整数': 'trainingResource.guaranteedGpuMustBeNonNegative',
    '保底 GPU 数不能超过最大 GPU 数': 'trainingResource.guaranteedGpuCannotExceedMax',
    '用户组最大 GPU 数不能超过资源池总容量': 'trainingResource.maxGpuCannotExceedPool',
    '资源池内各用户组保底 GPU 数之和不能超过资源池总容量': 'trainingResource.guaranteedGpuTotalCannotExceedPool',
    '训练预约不存在': 'trainingResource.reservationNotFound',
    '训练资源租约已过期': 'trainingResource.reservationExpired',
    '推理资源租约已过期': 'trainingResource.reservationExpired',
    '当前训练预约状态不允许续期': 'trainingResource.reservationRenewStatusInvalid',
    '用户名或密码错误': 'auth.loginFailed',
    'Username already exists': 'auth.usernameExists',
    'Current password is incorrect': 'auth.currentPasswordIncorrect',
    'User not found': 'auth.userNotFound',
    'Cannot modify your own administrator account': 'auth.cannotModifySelf',
    'Cannot remove the last administrator': 'auth.cannotRemoveLastAdmin',
    'Use logout to revoke your own current session': 'auth.useLogoutForSelf',
    'Administrator permission required': 'auth.adminOnlyAction',
    'Please log in first': 'auth.loginRequired',
};

export const translateAuthError = (
    error: unknown,
    t: Translate,
    fallbackKey: string,
) => {
    if (!(error instanceof Error)) {
        return t(fallbackKey);
    }

    const errorKey = error.message.match(/auth\.error\.[A-Za-z]+/)?.[0];
    const key = authErrorMessages[errorKey ?? error.message];
    if (key) {
        return t(key);
    }

    const poolNodeRequired = error.message.match(/^资源池节点 ([A-Za-z]+) 不能为空$/);
    if (poolNodeRequired) {
        return t('trainingResource.poolNodeFieldRequired', { field: poolNodeRequired[1] });
    }

    const poolNodeDuplicate = error.message.match(/^资源池节点 ([A-Za-z]+) 不能重复$/);
    if (poolNodeDuplicate) {
        return t('trainingResource.poolNodeFieldDuplicate', { field: poolNodeDuplicate[1] });
    }

    const poolConflict = error.message.match(/^节点已属于其他资源池：(.+)$/);
    if (poolConflict) {
        return t('trainingResource.nodeAlreadyInOtherPool', { nodes: poolConflict[1] });
    }

    const positiveInteger = error.message.match(/^(最大 GPU 数|最大并发任务数|单任务最大节点数)必须是正整数$/);
    if (positiveInteger) {
        const fieldKeys: Record<string, string> = {
            '最大 GPU 数': 'maxGpuField',
            '最大并发任务数': 'maxConcurrentJobsField',
            '单任务最大节点数': 'maxNodesPerJobField',
        };
        return t('trainingResource.mustBePositiveInteger', { field: t(`trainingResource.${fieldKeys[positiveInteger[1]]}`) });
    }

    return error.message || t(fallbackKey);
};

