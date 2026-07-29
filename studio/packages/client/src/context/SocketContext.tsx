import {
    createContext,
    ReactNode,
    useContext,
    useEffect,
    useState,
} from 'react';
import { io, Socket } from 'socket.io-client';

import { ClientConfig } from '@shared/config/client';
import { SocketEvents } from '@shared/types';
import { useAuth } from './AuthContext';

const SocketContext = createContext<Socket | null>(null);

export function SocketContextProvider({ children }: { children: ReactNode }) {
    const [socket, setSocket] = useState<Socket | null>(null);
    const { forceLogout } = useAuth();

    useEffect(() => {
        const newSocket = io(ClientConfig.socketPath, {
            withCredentials: true,
        });
        newSocket.on(SocketEvents.server.forceLogout, () => {
            forceLogout();
            newSocket.close();
        });
        setSocket(newSocket);

        return () => {
            newSocket.off(SocketEvents.server.forceLogout);
            newSocket.close();
        };
    }, [forceLogout]);

    return (
        <SocketContext.Provider value={socket}>
            {children}
        </SocketContext.Provider>
    );
}

export const useSocket = () => useContext(SocketContext);
