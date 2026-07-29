import { FormEvent, useState } from 'react';
import { KeyRoundIcon, LogOutIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useAuth } from '@/context/AuthContext';
import { translateAuthError } from '@/utils/authErrors';

const ForcePasswordChangePage = () => {
    const { t } = useTranslation();
    const { changePassword, logout, user } = useAuth();
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [submitting, setSubmitting] = useState(false);

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError('');

        if (newPassword !== confirmPassword) {
            setError(t('auth.passwordMismatch'));
            return;
        }

        try {
            setSubmitting(true);
            await changePassword(currentPassword, newPassword);
        } catch (changeError) {
            setError(
                translateAuthError(changeError, t, 'auth.changePasswordFailed'),
            );
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
                        <h1>{t('auth.forcePasswordChangeTitle')}</h1>
                        <p>
                            {t('auth.forcePasswordChangeDescription', {
                                username: user?.username ?? '',
                            })}
                        </p>
                    </div>
                </div>

                <form className="auth-form" onSubmit={handleSubmit}>
                    <label>
                        <span>{t('auth.currentPassword')}</span>
                        <div className="auth-input">
                            <KeyRoundIcon className="size-4" />
                            <Input
                                value={currentPassword}
                                onChange={(event) =>
                                    setCurrentPassword(event.target.value)
                                }
                                type="password"
                                autoComplete="current-password"
                                autoFocus
                                required
                            />
                        </div>
                    </label>
                    <label>
                        <span>{t('auth.newPassword')}</span>
                        <div className="auth-input">
                            <KeyRoundIcon className="size-4" />
                            <Input
                                value={newPassword}
                                onChange={(event) =>
                                    setNewPassword(event.target.value)
                                }
                                type="password"
                                autoComplete="new-password"
                                minLength={6}
                                required
                            />
                        </div>
                    </label>
                    <label>
                        <span>{t('auth.confirmNewPassword')}</span>
                        <div className="auth-input">
                            <KeyRoundIcon className="size-4" />
                            <Input
                                value={confirmPassword}
                                onChange={(event) =>
                                    setConfirmPassword(event.target.value)
                                }
                                type="password"
                                autoComplete="new-password"
                                minLength={6}
                                required
                            />
                        </div>
                    </label>

                    {error && <div className="auth-error">{error}</div>}

                    <Button type="submit" className="auth-submit" disabled={submitting}>
                        <KeyRoundIcon className="size-4" />
                        {submitting
                            ? t('auth.changingPassword')
                            : t('auth.forcePasswordChangeSubmit')}
                    </Button>
                    <Button type="button" variant="ghost" onClick={() => void logout()}>
                        <LogOutIcon className="size-4" />
                        {t('auth.logout')}
                    </Button>
                </form>
            </section>
        </main>
    );
};

export default ForcePasswordChangePage;
