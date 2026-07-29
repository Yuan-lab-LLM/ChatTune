import { createContext, ReactNode, useContext } from 'react';
import { message } from 'antd';
import { MessageInstance } from 'antd/lib/message/interface';

interface MessageApiContextType {
    messageApi: MessageInstance;
}

const MessageApiContext = createContext<MessageApiContextType | null>(null);

interface Props {
    children: ReactNode;
}

export function MessageApiContextProvider({ children }: Props) {
    const [messageApi, contextHolder] = message.useMessage();

    return (
        <MessageApiContext.Provider value={{ messageApi }}>
            {contextHolder}
            {children}
        </MessageApiContext.Provider>
    );
}

export function useMessageApi() {
    const context = useContext(MessageApiContext);
    if (!context) {
        throw new Error(
            'useMessageApi must be used within a MessageApiProvider',
        );
    }
    return context;
}
