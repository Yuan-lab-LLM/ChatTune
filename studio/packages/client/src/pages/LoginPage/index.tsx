import { FormEvent, useState } from 'react';
import { LockKeyholeIcon, LogInIcon, UserIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useAuth } from '@/context/AuthContext';
import { translateAuthError } from '@/utils/authErrors';

const LoginPage = () => {
    const { t } = useTranslation();
    const { login } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [submitting, setSubmitting] = useState(false);

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError('');
        setSubmitting(true);

        try {
            await login(username, password);
        } catch (loginError) {
            setError(translateAuthError(loginError, t, 'auth.loginFailed'));
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <main className="auth-page">
            <div className="auth-page-backdrop" aria-hidden="true" />
            <section className="auth-panel">
                <div className="auth-brand">
                    <div className="auth-brand-mark">M</div>
                    <div className="auth-brand-copy">
                        <h1>MedFlow ChatTune</h1>
                        <p>{t('auth.loginHint')}</p>
                    </div>
                </div>

                <form className="auth-form" onSubmit={handleSubmit}>
                    <label>
                        <span>{t('auth.username')}</span>
                        <div className="auth-input">
                            <UserIcon className="size-4" />
                            <Input
                                value={username}
                                onChange={(event) =>
                                    setUsername(event.target.value)
                                }
                                autoComplete="username"
                                autoFocus
                                required
                            />
                        </div>
                    </label>
                    <label>
                        <span>{t('auth.password')}</span>
                        <div className="auth-input">
                            <LockKeyholeIcon className="size-4" />
                            <Input
                                value={password}
                                onChange={(event) =>
                                    setPassword(event.target.value)
                                }
                                type="password"
                                autoComplete="current-password"
                                required
                            />
                        </div>
                    </label>

                    {error && <div className="auth-error">{error}</div>}

                    <Button type="submit" className="auth-submit" disabled={submitting}>
                        <LogInIcon className="size-4" />
                        {submitting ? t('auth.loggingIn') : t('auth.login')}
                    </Button>
                </form>
            </section>
        </main>
    );
};

export default LoginPage;
