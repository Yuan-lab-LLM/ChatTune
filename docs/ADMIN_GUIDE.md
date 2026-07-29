# Admin Guide

This guide covers administrator workflows in Studio.

## First-Time Setup Order

Open `STUDIO_PUBLIC_URL` in a browser and log in with the administrator account and the password generated in `runtime.env`.

After the first login, configure resources in this order:

1. Create ordinary users.
2. Create user groups and bind each group to a Runtime node.
3. Configure training, evaluation/inference, and GRPO containers for each user group.
4. Move ordinary users into the configured user groups.
5. Create GPU resource pools.
6. Add pool nodes and user-group quotas.
7. Confirm the pool is enabled and the user group has available quota.

Users cannot reliably run Agent tasks until their group has a Runtime node, task containers, and GPU pool quota.

## User Management

Administrators can:

- Create users with username, initial password, and role.
- Search and filter by username, role, user group, and online status.
- Move ordinary users between user groups.
- Enable or disable accounts.
- Reset passwords and revoke sessions.
- Delete users after confirming the impact.

Production deployments should keep at least two named administrator accounts. The initial administrator should be used for setup and emergency recovery only.

## User Groups and Docker Containers

Each ordinary user group needs:

- The Runtime node used by the group.
- Training container.
- Evaluation/inference container.
- GRPO/verl training container.
- Container revalidation after saving, to confirm existence and accessibility.

On the same Runtime node, different user groups should not bind the same container. Different Runtime nodes may use the same container name, but those containers should be independent.

## GPU Resource Pools

In the GPU resources tab:

1. Create a pool.
2. Add pool nodes.
3. Configure each node's `sshAlias`, `trainAddress`, allowed GPU indexes, and optional `ncclSocketIfname`.
4. Configure user-group quota: owning node, guaranteed GPU count, maximum GPU count, maximum concurrent jobs, and maximum nodes per job.
5. Review active reservations and handle expired or abnormal reservations.

Disabling a pool prevents new reservations but does not interrupt already running jobs. Pool and quota edits are locked while active reservations are `preparing`, `reserved`, or `running`.

## Sharing and Audit

- The sharing tab is used to approve or reject user resource publication requests, or for administrators to publish private resources directly.
- Before publishing, confirm that data, model artifacts, logs, and outputs do not contain personal information, restricted data, or secrets.
- The audit tab records administrator operations on users, groups, pools, sharing, and approvals. Keep database backups and export audit records according to deployment policy.
