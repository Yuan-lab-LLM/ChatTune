import findProcess from 'find-process';

export const checkProcessByPid = async (pid: number) => {
    try {
        const processes = await findProcess('pid', pid);
        return processes.length > 0;
    } catch (err) {
        console.error('Error checking process:', err);
        return false;
    }
};
