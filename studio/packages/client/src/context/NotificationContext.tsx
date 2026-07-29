import { createContext, ReactNode, useContext } from 'react';
import { notification } from 'antd';
import { NotificationInstance } from 'antd/lib/notification/interface';

interface NotificationContextType {
    notificationApi: NotificationInstance;
}

const NotificationContext = createContext<NotificationContextType | null>(null);

interface Props {
    children: ReactNode;
}

export function NotificationContextProvider({ children }: Props) {
    const [notificationApi, contextHolder] = notification.useNotification();

    return (
        <NotificationContext.Provider value={{ notificationApi }}>
            {contextHolder}
            {children}
        </NotificationContext.Provider>
    );
}

export function useNotification() {
    const context = useContext(NotificationContext);
    if (!context) {
        throw new Error(
            'useNotification must be used within a NotificationProvider',
        );
    }
    return context;
}
