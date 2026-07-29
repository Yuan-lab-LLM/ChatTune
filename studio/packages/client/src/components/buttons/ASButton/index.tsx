import { ReactNode, RefObject } from 'react';
import { Button, ButtonProps, Tooltip } from 'antd';
import { TooltipPlacement } from 'antd/lib/tooltip';

/**
 * Props for manager button component used in Dataset/Model/Evaluation managers.
 */
interface ManagerButtonProps extends ButtonProps {
    variant?: 'primary' | 'secondary' | 'ghost';
    size?: 'sm' | 'md' | 'lg';
    icon?: ReactNode;
    loading?: boolean;
}

/**
 * Unified button component for Dataset/Model/Evaluation managers.
 * Provides consistent styling across all managers.
 */
const ManagerButton = ({
    variant = 'secondary',
    size = 'sm',
    icon,
    children,
    className,
    ...restProps
}: ManagerButtonProps) => {
    const baseClasses = 'inline-flex items-center gap-1.5 rounded-xl font-medium transition-all duration-200 border';

    const variantClasses = {
        primary: 'bg-primary text-primary-foreground border-primary hover:bg-primary/90 shadow-sm',
        secondary: 'bg-background border-border/50 text-foreground hover:bg-muted hover:border-border',
        ghost: 'bg-transparent border-transparent text-muted-foreground hover:text-foreground hover:bg-muted/50'
    };

    const sizeClasses = {
        sm: 'px-3 py-1.5 text-xs',
        md: 'px-4 py-2 text-sm',
        lg: 'px-5 py-2.5 text-sm'
    };

    return (
        <Button
            className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className || ''}`}
            icon={icon}
            {...restProps}
        >
            {children}
        </Button>
    );
};

/**
 * Extended button props that include tooltip and placement configuration.
 */
interface Props extends ButtonProps {
    ref?: RefObject<null> | undefined;
    tooltip: string;
    placement?: TooltipPlacement;
}

/**
 * Secondary button with tooltip support and configurable placement.
 * Uses Ant Design's default type with minimal styling.
 */
const SecondaryButton = ({
    tooltip,
    placement = 'top',
    ...restProps
}: Props) => {
    return (
        <Tooltip title={tooltip} placement={placement}>
            <Button color="default" type="default" {...restProps} />
        </Tooltip>
    );
};

/**
 * Props for the switch button component that toggles between active/inactive states.
 */
interface SwitchButtonProps extends ButtonProps {
    tooltip: string;
    title?: string;
    activeIcon?: ReactNode;
    inactiveIcon?: ReactNode;
    active: boolean;
}

/**
 * Toggle button that switches between active and inactive states.
 * Changes background color, text color, and icon based on the `active` prop.
 */
const SwitchButton = ({
    tooltip,
    title,
    activeIcon,
    inactiveIcon,
    active,
    ...restProps
}: SwitchButtonProps) => {
    // Dynamic styling based on active state
    const bgColor = active ? 'var(--secondary)' : 'transparent';
    const color = active ? 'var(--secondary-foreground)' : 'var(--hint-color)';

    return (
        <Tooltip title={tooltip}>
            <Button
                style={{ background: bgColor, color: color }}
                icon={active ? activeIcon : inactiveIcon}
                className="as-switch-button"
                {...restProps}
            >
                {title}
            </Button>
        </Tooltip>
    );
};

export { SecondaryButton, SwitchButton, ManagerButton };
