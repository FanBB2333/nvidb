// Native UserNotifications sender bundled with nvidb's app icon.
#import <AppKit/AppKit.h>
#import <CoreServices/CoreServices.h>
#import <UserNotifications/UserNotifications.h>

@interface NotificationDelegate : NSObject <UNUserNotificationCenterDelegate>
@end

@implementation NotificationDelegate
- (void)userNotificationCenter:(UNUserNotificationCenter *)center
      willPresentNotification:(UNNotification *)notification
        withCompletionHandler:(void (^)(UNNotificationPresentationOptions))completion {
    completion(UNNotificationPresentationOptionBanner | UNNotificationPresentationOptionList);
}
@end

static void fail(NSString *message) {
    fprintf(stderr, "nvidb notification: %s\n", message.UTF8String);
    exit(1);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSArray<NSString *> *args = NSProcessInfo.processInfo.arguments;
        BOOL authorizeOnly = args.count == 2 && [args[1] isEqualToString:@"--authorize"];
        NSUInteger index = [args indexOfObject:@"--notify"];
        // Clicking a notification reopens the app without replaying the alert.
        if (index == NSNotFound && !authorizeOnly) return 0;
        if (!authorizeOnly && args.count != index + 4) return 2;

        // The helper lives in a hidden directory, outside the usual app scan.
        OSStatus status = LSRegisterURL((__bridge CFURLRef)NSBundle.mainBundle.bundleURL, true);
        if (status != noErr) fail(@"could not register the notification app");
        [NSApplication sharedApplication];
        UNMutableNotificationContent *content = [UNMutableNotificationContent new];
        if (!authorizeOnly) {
            content.title = args[index + 1];
            content.subtitle = args[index + 2];
            content.body = args[index + 3];
        }

        UNUserNotificationCenter *center = UNUserNotificationCenter.currentNotificationCenter;
        // UNUserNotificationCenter holds its delegate weakly.
        static NotificationDelegate *delegate;
        delegate = [NotificationDelegate new];
        center.delegate = delegate;
        // Interactive setup can wait for the user; normal alert delivery is bounded.
        int timeout = authorizeOnly ? 120 : 8;
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, timeout * NSEC_PER_SEC),
                       dispatch_get_main_queue(), ^{ fail(@"request timed out"); });
        [center requestAuthorizationWithOptions:UNAuthorizationOptionAlert
                              completionHandler:^(BOOL granted, NSError *error) {
            if (error) fail(error.localizedDescription);
            if (!granted) fail(@"notifications are disabled in System Settings");
            if (authorizeOnly) exit(0);
            UNNotificationRequest *request = [UNNotificationRequest
                requestWithIdentifier:NSUUID.UUID.UUIDString content:content trigger:nil];
            [center addNotificationRequest:request withCompletionHandler:^(NSError *error) {
                if (error) fail(error.localizedDescription);
                exit(0);
            }];
        }];
        dispatch_main();
    }
}
