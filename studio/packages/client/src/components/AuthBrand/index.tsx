interface AuthBrandProps {
    title: string;
    description: string;
    hideTitle?: boolean;
}

const AuthBrand = ({ title, description, hideTitle = false }: AuthBrandProps) => (
    <header className="auth-brand">
        <img className="auth-brand-logo" src="/logo.svg" alt="ChatTune" />
        <div className={hideTitle ? 'auth-brand-copy auth-brand-copy--title-hidden' : 'auth-brand-copy'}>
            <h1 className={hideTitle ? 'sr-only' : undefined}>{title}</h1>
            <p>{description}</p>
        </div>
    </header>
);

export default AuthBrand;
